import cv2
import numpy as np


class ImageProcessor:
    def __init__(self) -> None:
        self._concave_point_indices = []


    def midpoint(self, p1, p2):
        return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)


    def get_centroid(c):
        M = cv2.moments(np.array(c))
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return cx, cy


    def merge_contour(c1, c2):
        return np.concatenate((np.array(c1).reshape(-1,1,2), np.array(c2).reshape(-1,1,2)))


    def get_concave_points(self, image, blur_kernel=11):
        # preprocess
        blurred = cv2.medianBlur(image, blur_kernel)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # get contours
        countours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = sorted(countours, key=cv2.contourArea)[-2]    # TODO: [-2] because the biggest is the corner

        # get approximated contour
        peri = cv2.arcLength(biggest_contour, True)
        approx = cv2.approxPolyDP(biggest_contour, 0.01*peri, True)

        # get concave points
        concave_points = []
        for n, poi in enumerate(approx):
            if n != len(approx)-1:
                midpt = self.midpoint(approx[n-1][0], approx[n+1][0])
            else:
                midpt = self.midpoint(approx[n-1][0], approx[0][0])

            dist = cv2.pointPolygonTest(biggest_contour, midpt, False)
            if dist < 0:
                concave_points.append(approx[n])
        

        for i, p in enumerate(biggest_contour):
            if p[0].tolist() in np.array(concave_points).reshape(-1, 2).tolist():
                self._concave_point_indices.append(i)

        return self._concave_point_indices
    

    def calculate_ellipticity(self, cnt1, cnt2=None):
        if cnt2 is not None:
            pass # merge contour
        else:
            cnt2process = np.array(cnt1).reshape(-1,2)

        ((centx,centy), axes, angle) = cv2.fitEllipse(cnt2process)
        # major and minor (a & b)
        major_length = max(axes)
        minor_length = min(axes)

        # get angle
        if angle > 90:
            angle = 90 + angle
        elif angle <= 90:
            angle = 90 - angle

        
        sigma_ADDi = 0
        for p in cnt2process:
            xi = p[0]
            yi = p[1]
            
            # transform
            xi_ = (np.cos(angle)*(xi-int(centx))) - (np.sin(angle)*(yi-int(centy)))
            yi_ = (np.sin(angle)*(xi-int(centx))) + (np.cos(angle)*(yi-int(centy)))

            Di = np.sqrt((xi_**2) / (major_length**2) + (yi_**2) / (minor_length**2))
            ADDi = (np.sqrt((xi_**2)+(yi_**2))) * (1 - (1/np.abs(Di)))
            sigma_ADDi += ADDi

        return np.abs(sigma_ADDi/len(cnt2process))/100
    

    def calculate_concavity(self, cnt1, cnt2=None):
        if cnt2 is not None:
            cnt_ = self.merge_contour(cnt1, cnt2)
            cnt_ = np.array(cnt_).reshape(-1,1,2)
        else:
            cnt_ = np.array(cnt1).reshape(-1,1,2)
        
        hull = cv2.convexHull(cnt_)

        peri = cv2.arcLength(cnt_, True)
        approx_ = cv2.approxPolyDP(cnt_, 0.015*peri, True)

        # cek if theres any concave point?
        concave_points_ = list()
        for n, poi in enumerate(approx_):
            if n != len(approx_)-1:
                midpt = self.midpoint(approx_[n-1][0], approx_[n+1][0])
                dist = cv2.pointPolygonTest(cnt_, midpt, False)
                if dist < 0:    
                    concave_points_.append(approx_[n])
            else:
                midpt = self.midpoint(approx_[n-1][0], approx_[0][0])
                dist = cv2.pointPolygonTest(cnt_, midpt, False)
                if dist < 0:
                    concave_points_.append(approx_[n])

        if len(concave_points_) == 0:
            return (1-cv2.contourArea(cnt_)/cv2.contourArea(hull))
        else:
            return 1
        