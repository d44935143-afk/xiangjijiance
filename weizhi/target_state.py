def to_plain_list(value):
    """Convert numpy arrays to plain lists for JSON output."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def build_checkerboard_info(region):
    """Build serializable checkerboard calibration information."""
    board = getattr(region, "checkerboard", None)
    if board is None:
        return {
            "found": False,
            "fit_error": None,
            "center": None,
            "outer_corners": None,
            "inner_corners": None,
            "homography_pixel_to_mm": None,
            "homography_mm_to_pixel": None,
            "metrics": None,
        }

    return {
        "found": True,
        "fit_error": board.get("fit_error"),
        "center": to_plain_list(board.get("center_global", board.get("center"))),
        "outer_corners": to_plain_list(
            board.get("outer_corners_global", board.get("outer_corners"))
        ),
        "inner_corners": to_plain_list(
            board.get("inner_corners_global", board.get("inner_corners"))
        ),
        "homography_pixel_to_mm": to_plain_list(
            getattr(region, "homography_pixel_to_mm", None)
        ),
        "homography_mm_to_pixel": to_plain_list(
            getattr(region, "homography_mm_to_pixel", None)
        ),
        "metrics": board.get("metrics"),
    }


def build_digit_info(region, number):
    """Build serializable digit-recognition information."""
    return {
        "id": number,
        "score": region.digit_score,
        "iou_score": region.digit_iou_score,
        "shape_score": region.digit_shape_score,
        "bbox": region.digit_bbox,
        "rectified_bbox": getattr(region, "rectified_digit_bbox", None),
        "rectified_crop_shape": to_plain_list(
            getattr(region, "rectified_crop_shape", None)
        ),
        "homography_crop_to_front": to_plain_list(
            getattr(region, "homography_crop_to_front", None)
        ),
    }


def build_targets_dict(results):
    """Create the saved target-state dictionary from recognition results."""
    targets = {}

    for region, number in results:
        if number is None:
            continue

        number = str(number)
        targets[number] = {
            "number": number,
            "template_id": region.match_template_id,
            "initial_center": region.center,
            "current_center": region.center,
            "rect": region.rect,
            "match_score": region.match_score,
            "digit_score": region.digit_score,
            "digit": build_digit_info(region, number),
            "checkerboard": build_checkerboard_info(region),
            "settlement_mm": 0.0,
            "lost": False,
        }

    return targets


def update_target(targets, number, rect):
    """Update one target rectangle and current center."""
    number = str(number)

    if number not in targets:
        return

    x, y, w, h = rect
    center = (x + w / 2.0, y + h / 2.0)

    targets[number]["rect"] = rect
    targets[number]["current_center"] = center
    targets[number]["lost"] = False


def mark_target_lost(targets, number):
    """Mark one target as lost."""
    number = str(number)

    if number in targets:
        targets[number]["lost"] = True


def update_settlement(targets, number, scale_y=1.0, settlement_sign=1):
    """Update settlement in millimeters from the vertical pixel displacement."""
    number = str(number)

    if number not in targets:
        return None

    initial_center = targets[number]["initial_center"]
    current_center = targets[number]["current_center"]

    dy_pixel = current_center[1] - initial_center[1]
    settlement_mm = dy_pixel * scale_y * settlement_sign

    targets[number]["settlement_mm"] = settlement_mm
    return settlement_mm


def get_target(targets, number):
    """Return one target entry by number."""
    return targets.get(str(number))


def get_all_targets(targets):
    """Return all target entries."""
    return targets
