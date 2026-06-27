import torch
import torchvision.ops as ops

def slice_image_tensor(image_tensor, slice_wh=(320, 320), overlap_ratio=0.2):
    """
    Slices a high-resolution input image tensor into a grid of overlapping patches.
    Input shape: [C, H, W]
    """
    _, h, w = image_tensor.shape
    slice_w, slice_h = slice_wh

    step_x = int(slice_w * (1 - overlap_ratio))
    step_y = int(slice_h * (1 - overlap_ratio))

    slices = []
    starting_boxes = [] # Tracking the (x_min, y_min) offset to reconstruct global coordinates

    y = 0
    while y < h:
        x = 0
        while x < w:
            # Handle boundary conditions near the edges
            y_end = min(y + slice_h, h)
            x_end = min(x + slice_w, w)
            y_start = max(0, y_end - slice_h)
            x_start = max(0, x_end - slice_w)

            crop = image_tensor[:, y_start:y_end, x_start:x_end]
            slices.append(crop)
            starting_boxes.append([x_start, y_start])

            if x_end == w: break
            x += step_x
        if y_end == h: break
        y += step_y

    return torch.stack(slices), starting_boxes

def sahi_fod_inference(model, image_tensor, device, confidence_threshold=0.35, nms_iou_threshold=0.45):
    """
    Executes Slicing Aided Inference exclusively via the specialized Phase 2 FOD Head.
    """
    model.eval()
    with torch.no_grad():
        # 1. Break the input frame down into high-detail patches
        patches, offsets = slice_image_tensor(image_tensor, slice_wh=(320, 320))
        patches = patches.to(device)

        # 2. Run an inference pass over all patches simultaneously
        # Output shape format assumed from Head 3: [Patches, 4_coords + 31 classes, H_grid, W_grid]
        preds = model(patches, task_id=2)["fod"]

        # Collapse spatial grid dimensions to extract raw prediction arrays
        preds = preds.mean(dim=[2, 3])

        global_boxes = []
        global_scores = []
        global_classes = []

        for idx, offset in enumerate(offsets):
            pred_slice = preds[idx]
            bbox_delta = pred_slice[:4]
            class_logits = pred_slice[4:]

            scores = torch.softmax(class_logits, dim=0)
            best_score, best_class = torch.max(scores, dim=0)

            if best_score > confidence_threshold:
                # Map localized patch coordinates back onto the global frame coordinates
                x_offset, y_offset = offset
                global_x = bbox_delta[0] + x_offset
                global_y = bbox_delta[1] + y_offset
                w = bbox_delta[2]
                h = bbox_delta[3]

                # Convert from center coordinates (cx, cy, w, h) to corner coordinates (x1, y1, x2, y2)
                x1 = global_x - (w / 2)
                y1 = global_y - (h / 2)
                x2 = global_x + (w / 2)
                y2 = global_y + (h / 2)

                global_boxes.append([x1, y1, x2, y2])
                global_scores.append(best_score)
                global_classes.append(best_class)

        if len(global_boxes) == 0:
            return torch.empty((0, 4)), torch.empty((0)), torch.empty((0))
        
        # Convert lists to PyTorch tensors
        boxes_tensor = torch.tensor(global_boxes, device=device)
        scores_tensor = torch.tensor(global_scores, device=device)
        classes_tensor = torch.tensor(global_classes, device=device)

        # 3. Apply Non-Maximum Suppresion (NMS) to eliminate overlapping duplicate boxes
        keep_indices = ops.nms(boxes_tensor, scores_tensor, nms_iou_threshold)

        return boxes_tensor[keep_indices], scores_tensor[keep_indices], classes_tensor[keep_indices]

