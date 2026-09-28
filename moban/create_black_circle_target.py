from pathlib import Path

from PIL import Image, ImageDraw


DPI = 300
CM_TO_INCH = 1 / 2.54
TARGET_SIZE_CM = 3
GRID_SIZE = 2


def cm_to_px(value):
    return int(round(value * CM_TO_INCH * DPI))


def create_checkerboard_target():
    size_px = cm_to_px(TARGET_SIZE_CM)
    image = Image.new("RGB", (size_px, size_px), "white")
    draw = ImageDraw.Draw(image)
    cell_px = size_px / GRID_SIZE

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            color = "black" if (row + col) % 2 == 0 else "white"
            x0 = round(col * cell_px)
            y0 = round(row * cell_px)
            x1 = round((col + 1) * cell_px) - 1
            y1 = round((row + 1) * cell_px) - 1
            draw.rectangle([x0, y0, x1, y1], fill=color)

    return image


def main():
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "checkerboard_2x2_3cm.png"

    image = create_checkerboard_target()
    image.save(output_path, dpi=(DPI, DPI))

    print("Generated:", output_path)


if __name__ == "__main__":
    main()
