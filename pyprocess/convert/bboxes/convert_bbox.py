def validate_numeric(*args):
    """
    Tüm argümanların sayısal (int veya float) olup olmadığını kontrol eder.

    Args:
        *args: Kontrol edilecek argümanlar.

    Raises:
        TypeError: Eğer herhangi bir argüman sayısal değilse.
    """
    if not all(isinstance(arg, (int, float)) for arg in args):
        raise TypeError("All arguments must be numeric.")

def validate_size(size):
    """
    Verilen boyutun geçerli olup olmadığını kontrol eder.

    Args:
        size: Kontrol edilecek boyut (list veya tuple).

    Raises:
        TypeError: Eğer size list veya tuple değilse veya uzunluğu 2 değilse.
        ValueError: Eğer boyutlar pozitif sayılar değilse.
    """

    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise TypeError("Size must be a list or tuple of length 2.")
    if not all(isinstance(dim, (int, float)) and dim > 0 for dim in size):
        raise ValueError("Image dimensions must be positive numbers.")

def validate_non_negative(*args):
    """
    Tüm argümanların negatif olmadığını kontrol eder.

    Args:
        *args: Kontrol edilecek argümanlar.

    Raises:
        ValueError: Eğer herhangi bir argüman negatifse.
    """
    if any(arg < 0 for arg in args):
        raise ValueError("All arguments must be non-negative.")

def pascal_to_coco(x_min, y_min, x_max, y_max):
    """
    Pascal VOC formatından COCO formatına dönüşüm yapar.

    Args:
        x_min (float): Sol üst köşenin x koordinatı.
        y_min (float): Sol üst köşenin y koordinatı.
        x_max (float): Sağ alt köşenin x koordinatı.
        y_max (float): Sağ alt köşenin y koordinatı.

    Returns:
        list: [x, y, width, height] formatında COCO bounding box.

    Raises:
        ValueError: Eğer koordinatlar geçersizse.
    """
    validate_numeric(x_min, y_min, x_max, y_max)
    validate_non_negative(x_min, y_min, x_max, y_max)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid Pascal bounding box coordinates.")
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def pascal_to_yolo(x_min, y_min, x_max, y_max, size):
    """
    Pascal VOC formatından YOLO formatına dönüşüm yapar.

    Args:
        x_min (float): Sol üst köşenin x koordinatı.
        y_min (float): Sol üst köşenin y koordinatı.
        x_max (float): Sağ alt köşenin x koordinatı.
        y_max (float): Sağ alt köşenin y koordinatı.
        size (tuple): Görüntü boyutu (width, height).

    Returns:
        list: [x_center, y_center, width, height] formatında normalize edilmiş YOLO bounding box.

    Raises:
        ValueError: Eğer koordinatlar geçersizse veya görüntü boyutları dışındaysa.
    """

    validate_numeric(x_min, y_min, x_max, y_max)
    validate_size(size)
    validate_non_negative(x_min, y_min, x_max, y_max)
    max_width, max_height = size
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid Pascal bounding box coordinates.")
    if x_min < 0 or y_min < 0 or x_max > max_width or y_max > max_height:
        raise ValueError("Bounding box coordinates are outside image dimensions.")
    x_center = ((x_min + x_max) / 2) / max_width
    y_center = ((y_min + y_max) / 2) / max_height
    r_width = (x_max - x_min) / max_width
    r_height = (y_max - y_min) / max_height
    return [x_center, y_center, r_width, r_height]

def coco_to_pascal(x_min, y_min, width, height):
    """
    COCO formatından Pascal VOC formatına dönüşüm yapar.

    Args:
        x_min (float): Sol üst köşenin x koordinatı.
        y_min (float): Sol üst köşenin y koordinatı.
        width (float): Bounding box'ın genişliği.
        height (float): Bounding box'ın yüksekliği.

    Returns:
        list: [x_min, y_min, x_max, y_max] formatında Pascal VOC bounding box.

    Raises:
        ValueError: Eğer koordinatlar veya boyutlar geçersizse.
    """

    validate_numeric(x_min, y_min, width, height)
    validate_non_negative(x_min, y_min, width, height)
    if x_min < 0 or y_min < 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid COCO bounding box parameters.")
    return [x_min, y_min, x_min + width, y_min + height]

def coco_to_yolo(x_min, y_min, width, height, size):
    """
    COCO formatından YOLO formatına dönüşüm yapar.

    Args:
        x_min (float): Sol üst köşenin x koordinatı.
        y_min (float): Sol üst köşenin y koordinatı.
        width (float): Bounding box'ın genişliği.
        height (float): Bounding box'ın yüksekliği.
        size (tuple): Görüntü boyutu (width, height).

    Returns:
        list: [x_center, y_center, width, height] formatında normalize edilmiş YOLO bounding box.

    Raises:
        ValueError: Eğer koordinatlar veya boyutlar geçersizse veya görüntü boyutları dışındaysa.
    """

    validate_numeric(x_min, y_min, width, height)
    validate_size(size)
    validate_non_negative(x_min, y_min, width, height)
    max_width, max_height = size
    if x_min < 0 or y_min < 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid COCO bounding box parameters.")
    if x_min + width > max_width or y_min + height > max_height:
        raise ValueError("Bounding box extends beyond image dimensions.")
    x_center = (x_min + (width / 2)) / max_width
    y_center = (y_min + (height / 2)) / max_height
    r_width = width / max_width
    r_height = height / max_height
    return [x_center, y_center, r_width, r_height]

def yolo_to_coco(x_center, y_center, r_width, r_height, size):
    """
    YOLO formatından COCO formatına dönüşüm yapar.

    Args:
        x_center (float): Normalize edilmiş merkez x koordinatı.
        y_center (float): Normalize edilmiş merkez y koordinatı.
        r_width (float): Normalize edilmiş genişlik.
        r_height (float): Normalize edilmiş yükseklik.
        size (tuple): Görüntü boyutu (width, height).

    Returns:
        list: [x_min, y_min, width, height] formatında COCO bounding box.

    Raises:
        ValueError: Eğer normalize edilmiş koordinatlar veya boyutlar geçersizse.
    """

    validate_numeric(x_center, y_center, r_width, r_height)
    validate_size(size)
    max_width, max_height = size
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
        raise ValueError("Center coordinates must be between 0 and 1.")
    if not (0 < r_width <= 1 and 0 < r_height <= 1):
        raise ValueError("Invalid YOLO bounding box dimensions.")
    width = r_width * max_width
    height = r_height * max_height
    x_min = (x_center * max_width) - (width / 2)
    y_min = (y_center * max_height) - (height / 2)
    return [x_min, y_min, width, height]

def yolo_to_pascal(x_center, y_center, r_width, r_height, size):
    """
    YOLO formatından Pascal VOC formatına dönüşüm yapar.

    Args:
        x_center (float): Normalize edilmiş merkez x koordinatı.
        y_center (float): Normalize edilmiş merkez y koordinatı.
        r_width (float): Normalize edilmiş genişlik.
        r_height (float): Normalize edilmiş yükseklik.
        size (tuple): Görüntü boyutu (width, height).

    Returns:
        list: [x_min, y_min, x_max, y_max] formatında Pascal VOC bounding box.

    Raises:
        ValueError: Eğer normalize edilmiş koordinatlar veya boyutlar geçersizse.
    """

    validate_numeric(x_center, y_center, r_width, r_height)
    validate_size(size)
    max_width, max_height = size
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
        raise ValueError("Center coordinates must be between 0 and 1.")
    if not (0 < r_width <= 1 and 0 < r_height <= 1):
        raise ValueError("Invalid YOLO bounding box dimensions.")
    width = r_width * max_width
    height = r_height * max_height
    x_min = (x_center * max_width) - (width / 2)
    y_min = (y_center * max_height) - (height / 2)
    x_max = x_min + width
    y_max = y_min + height
    return [x_min, y_min, x_max, y_max]

def convert_bbox(from_format, to_format, *args, **kwargs):
    """
    Bounding box formatları arasında dönüşüm yapar.

    Args:
        from_format (str): Kaynak format ('pascal', 'coco', veya 'yolo').
        to_format (str): Hedef format ('pascal', 'coco', veya 'yolo').
        *args: Bounding box koordinatları.
        **kwargs: Ek parametreler (örn. 'size').

    Returns:
        list: Dönüştürülmüş bounding box koordinatları.

    Raises:
        ValueError: Desteklenmeyen bir dönüşüm istendiğinde.
    """
    func_name = f"{from_format}_to_{to_format}"
    conversion_func = globals().get(func_name)
    if conversion_func:
        return conversion_func(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported conversion: {from_format} to {to_format}")

def main():
    # Test pascal_to_coco
    print("Testing pascal_to_coco:")
    try:
        print(convert_bbox('pascal', 'coco', 50, 50, 150, 150))  # Expected: [50, 50, 100, 100]
        print(convert_bbox('pascal', 'coco', 0, 0, 200, 100))    # Expected: [0, 0, 200, 100]
        print(convert_bbox('pascal', 'coco', -10, -10, 50, 50))  # Expected: ValueError
    except ValueError as e:
        print(f"Error: {e}")

    # Test pascal_to_yolo
    print("\nTesting pascal_to_yolo:")
    try:
        print(convert_bbox('pascal', 'yolo', 50, 50, 150, 150, size=(200, 200)))  # Expected: [0.5, 0.5, 0.5, 0.5]
        print(convert_bbox('pascal', 'yolo', 0, 0, 200, 100, size=(200, 200)))    # Expected: [0.5, 0.25, 1.0, 0.5]
        print(convert_bbox('pascal', 'yolo', 0, 0, 250, 100, size=(200, 200)))    # Expected: ValueError
    except ValueError as e:
        print(f"Error: {e}")

    # Test coco_to_pascal
    print("\nTesting coco_to_pascal:")
    try:
        print(convert_bbox('coco', 'pascal', 50, 50, 100, 100))  # Expected: [50, 50, 150, 150]
        print(convert_bbox('coco', 'pascal', 0, 0, 200, 100))    # Expected: [0, 0, 200, 100]
        print(convert_bbox('coco', 'pascal', -10, -10, 50, 50))  # Expected: ValueError
    except ValueError as e:
        print(f"Error: {e}")

    # Test coco_to_yolo
    print("\nTesting coco_to_yolo:")
    try:
        print(convert_bbox('coco', 'yolo', 50, 50, 100, 100, size=(200, 200)))  # Expected: [0.5, 0.5, 0.5, 0.5]
        print(convert_bbox('coco', 'yolo', 0, 0, 200, 100, size=(200, 200)))    # Expected: [0.5, 0.25, 1.0, 0.5]
        print(convert_bbox('coco', 'yolo', 0, 0, 250, 100, size=(200, 200)))    # Expected: ValueError
    except ValueError as e:
        print(f"Error: {e}")

    # Test yolo_to_coco
    print("\nTesting yolo_to_coco:")
    try:
        print(convert_bbox('yolo', 'coco', 0.5, 0.5, 0.5, 0.5, size=(200, 200)))  # Expected: [50, 50, 100, 100]
        print(convert_bbox('yolo', 'coco', 0.5, 0.25, 1.0, 0.5, size=(200, 200))) # Expected: [0, 0, 200, 100]
        print(convert_bbox('yolo', 'coco', 1.1, 0.5, 0.5, 0.5, size=(200, 200)))  # Expected: ValueError
    except ValueError as e:
        print(f"Error: {e}")

    # Test yolo_to_pascal
    print("\nTesting yolo_to_pascal:")
    try:
        print(convert_bbox('yolo', 'pascal', 0.5, 0.5, 0.5, 0.5, size=(200, 200)))  # Expected: [50, 50, 150, 150]
        print(convert_bbox('yolo', 'pascal', 0.5, 0.25, 1.0, 0.5, size=(200, 200))) # Expected: [0, 0, 200, 100]
        print(convert_bbox('yolo', 'pascal', 1.1, 0.5, 0.5, 0.5, size=(200, 200)))  # Expected: ValueError
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()