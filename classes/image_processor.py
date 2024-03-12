import cv2

class ImageProcessor:
    def __init__(self) -> None:
        self.__concave_points = list()

    def midpoint(self, p1, p2):
        return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

    def get_concave_points(self, image, blur_kernel=11):
        blurred = cv2.medianBlur(image, blur_kernel)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        countours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(countours, key=cv2.contourArea)

        peri = cv2.arcLength(c[-2], True)
        approx = cv2.approxPolyDP(c[-2], 0.01*peri, True)

        for n, poi in enumerate(approx):
            if n != len(approx)-1:
                midpt = self.midpoint(approx[n-1][0], approx[n+1][0])
                dist = cv2.pointPolygonTest(c[-2], midpt, False)
                if dist < 0:    
                    self.__concave_points.append(approx[n])
            else:
                midpt = self.midpoint(approx[n-1][0], approx[0][0])
                dist = cv2.pointPolygonTest(c[-2], midpt, False)
                if dist < 0:
                    self.__concave_points.append(approx[n])
        
        return self.__concave_points

