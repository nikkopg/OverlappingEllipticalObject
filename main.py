from classes.image_processor import ImageProcessor
import logging
import cv2

def main():
    image = cv2.imread("input/dummy.jpg", 0)
    processor = ImageProcessor()

    biggest_contour = processor.get_biggest_contour(image)
    concave_point_indices = processor.get_concave_points(biggest_contour)
    contour_segments = processor.get_contour_segments(biggest_contour, concave_point_indices)
    print(f"Found {len(contour_segments)} segments.")

if __name__ == '__main__':
    main()