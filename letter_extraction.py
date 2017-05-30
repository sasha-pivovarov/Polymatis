#encoding: utf-8
#from wand.image import Image, Color
#from PIL import Image
import subprocess
import cv2
import io
import numpy as np
import random
from math import fabs
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering, MeanShift
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, NMF
from os import listdir, makedirs, path


class LetterDetection:

    detector = None
    struct_element = None
    clustering = KMeans(n_clusters=50)
    selection = None
    sColor = None
    sSpace = None
    clahe = cv2.createCLAHE(2.0, (4, 4))
    special_chars = {"dot": "·"}
    #image_dump_path = None

    def __init__(self, areafilter, selement_h, selement_w, scolor, sspace, areamin=0, areamax=5000, cluster_algo="DBSCAN", ntraits="mle"):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = areafilter
        if areafilter:
            params.minArea = areamin
            params.maxArea = areamax
        self.detector = cv2.SimpleBlobDetector_create()
        self.erosion_element = cv2.getStructuringElement(cv2.MORPH_RECT, (selement_h,selement_w))
        self.selection = NMF(n_components=180, init="nndsvda")
        self.sColor = scolor
        self.sSpace = sspace
        #self.image_dump_path = image_path
        if cluster_algo == "KMeans":
            self.clustering = KMeans(n_clusters=40)
        elif cluster_algo == "Agglomerative":
            self.clustering = AgglomerativeClustering(n_clusters=40)
        elif cluster_algo =="Affinity":
            self.clustering == AffinityPropagation()
        elif cluster_algo == "DBSCAN":
            self.clustering == DBSCAN()


    def pdf_to_jpg_stream(self, pdfpath, start, end):
        foldername = path.splitext(pdfpath)[0].split('/')[-1]
        dumppath = ("pngs/" + foldername + '/')
        if not path.exists(dumppath):
            makedirs(dumppath)
            subprocess.call(["pdfimages", "-png", pdfpath, dumppath])


        for y in listdir(dumppath):
            try:
                ynum = int((path.splitext(y)[0].split('-')[1].lstrip("0")))
            except ValueError:
                ynum = 0
            if path.isfile(dumppath + y) and ynum in range(start, end):
                yield dumppath + y


    def resize_page(self, page):
        height, width = page.shape[:2]
        ratio = 700 / float(height)
        return cv2.resize(page, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_AREA)


    def transform_page(self, page, block_size, C):
        image = cv2.imread(page, cv2.IMREAD_COLOR)
        image = self.resize_page(image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image_gray = self.clahe.apply(image_gray)
        #image_gray = cv2.equalizeHist(image_gray)
        smoothed_image = cv2.bilateralFilter(image_gray, -1, self.sColor, self.sSpace)
        #thresholded_image = cv2.adaptiveThreshold(smoothed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
        ret, thresholded_image = cv2.threshold(smoothed_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        return thresholded_image


    def get_bounding_box(self, image, skewCorrection=False, verbose=False):
        #display_image = image.copy()
        eroded_image = cv2.erode(image, self.erosion_element)
        if verbose:
            cv2.imshow("Dots", eroded_image)
            cv2.waitKey()

        points = cv2.findNonZero(eroded_image)
        try:
            rect = cv2.minAreaRect(points)
            # box = cv2.boxPoints(rect)
            # box = np.int64(box)
            angle = rect[2]
            size = rect[1]

            if angle < -45:
                angle += 90
                size = tuple([size[1], size[0]])

            tocrop = image
            if skewCorrection:
                rotation_matrix = cv2.getRotationMatrix2D(tuple([int(round(x)) for x in rect[0]]), angle, 1)
                tocrop = cv2.warpAffine(image, rotation_matrix, image.shape[:2], flags=cv2.INTER_NEAREST)

            ret, cropped_image = cv2.threshold(cv2.getRectSubPix(tocrop, tuple([int(round(x)) for x in size]), tuple([int(round(x)) for x in rect[0]])), thresh=1, type=cv2.THRESH_BINARY, maxval=255)

        except cv2.error:
            return None

        return cropped_image

        #cv2.rectangle(display_image, (box[1, 0], box[1, 1]), (box[3, 0], box[3, 1]), (255, 0, 0), 2)
        #img_with_box = cv2.drawKeypoints(eroded_image, box.points, np.array([]), flags = cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        #cv2.imshow(str(angle), display_image)
        #cv2.imwrite("test_img%d.jpg" %box[1,1], display_image)
        #cv2.waitKey()

        #cv2.imshow("img", img_with_box)


    def get_contours(self, image):
        im, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        return im, contours, hierarchy


    def cut_contours(self, image, contours):
        for contour in contours:
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED, cv2.LINE_8)  #(mask, [contour], -1, 255, -1, cv2.FILLED)
            out = np.zeros_like(image)
            out[mask==255] = image[mask==255]
            points = cv2.findNonZero(out)

            try:
                bound_box = cv2.minAreaRect(points)
                angle = bound_box[2]
                size = bound_box[1]
                if angle < -45:
                    angle += 90
                    size = tuple([size[1], size[0]])

                ret, cropped_image = cv2.threshold(cv2.getRectSubPix(out, tuple([int(round(x)) for x in size]),
                                                                         tuple([int(round(x)) for x in bound_box[0]])),
                                                   thresh=1, type=cv2.THRESH_BINARY, maxval=255)

                assert size[0] > 0 and size[1] > 0 and len(points) > 40
                yield  cropped_image

            except (AssertionError, cv2.error):
                continue

    def get_text_region_bounds(self, bin_image):
        eroded_image = cv2.erode(bin_image, self.erosion_element)
        points = cv2.findNonZero(eroded_image)
        return cv2.boundingRect(points)


    def boxes_to_spaces(self, boxes):
        spaces = []
        left_coords = [x[0] for x in boxes]
        for box in boxes:
            own_right = box[1] + box[3]
            try:
                next_left = min([x for x in left_coords if x > own_right])
                spaces.append(next_left - own_right)
            except ValueError:
                continue

        return spaces


    def get_spacing_distribution(self, pdf, verbose):

        crop = None
        answered = False
        avg_color = None

        while not answered:
            for page in self.pdf_to_jpg_stream(pdf, 40, 50):
                bin_page = self.transform_page(page, 9, 10)
                x, y, w, h = self.get_text_region_bounds(bin_page)
                crop = bin_page[y: (y + h), x: (x + w)]
                cv2.imshow("Is this image cropped correctly?", crop)
                cv2.waitKey()
                answer = input("Y|N :  ").lower()
                if answer == "n":
                    continue
                elif answer != "y":
                    print("Please answer one of Y|N.")
                else:
                    avg_color = self.get_mean_color(page)
                    answered = True
                    break

        if verbose:
            swatch = np.array([[[avg_color] * 100] * 100], np.uint8)
            cv2.imshow("Average text color", swatch)
            cv2.waitKey()
        im, contours, hierarchy = self.get_contours(crop)
        boxes = [cv2.boundingRect(x) for x in contours]
        lines = []
        spaces = []
        for box in boxes:

            #box_xrange = box[0], box[0] + box[2]
            box_yrange = box[1], box[1] + box[3]
            if not [(box_yrange[0] in range(x[0], x[1]) or box_yrange[1] in range(x[0], x[1])) for x in [y[0] for y in lines]].__contains__(True):
                line_boxes = [linebox for linebox in boxes if linebox[1] in range(box_yrange[0], box_yrange[1]) or (linebox[1] + linebox[3]) in range(box_yrange[0], box_yrange[1])]
                lines.append((min([x[1] for x in line_boxes]), max((x[1] + x[3]) for x in line_boxes)))
                spaces.extend(self.boxes_to_spaces(line_boxes))

        return np.mean(spaces), np.std(spaces), avg_color

    def get_mean_color(self, page):
        return np.uint8(np.average(np.average(page[:, 0:10], axis=0), axis=0))

    def generate_phrases(self, pdf, text, verbose=False):
        spacing_distribution = (3, 2, (118, 166, 196))
        pdf_name = path.splitext(pdf)[0].split('/')[-1]
        symbols = {x : [cv2.imread("symbol_shapes/" + pdf_name + "/"+ x + "/" + y, cv2.IMREAD_COLOR) for y in listdir("symbol_shapes/" + pdf_name + "/" + x)] for x in listdir("symbol_shapes/" + pdf_name)}
        for key, value in self.special_chars.items():
            if key in symbols.keys():
                symbols[value] = symbols[key]

        text_lines = [x.strip() for x in open("texts/" + text, encoding="utf-8").readlines()]
        #imgs = []
        counter = 0
        for line in text_lines:
            #line = line.decode("utf-8")
            img = None
            for i in range(len(line)):
                if i == 0:
                    img = random.choice(symbols[line[0]])
                    continue
                space_size = int(fabs(np.random.normal(spacing_distribution[0], spacing_distribution[1])))
                cur_img_height = img.shape[0]
                difs = [cur_img_height - x.shape[0] for x in symbols[line[i]]]
                abs_difs = [fabs(x) for x in difs]
                abs_dif = min(abs_difs)
                cur_index = abs_difs.index(abs_dif)
                cur_char = symbols[line[i]][cur_index]
                cur_dif = difs[cur_index]
                print(img.shape)
                print(cur_char.shape)
                print(abs_dif)
                print(space_size)

                #if abs_dif % 2 != 0: abs_dif += 1
                if cur_dif < 0:
                    left = cv2.copyMakeBorder(img, top=int(abs_dif), left=0, right=int(space_size), bottom=0, borderType=cv2.BORDER_CONSTANT,
                                                       value=spacing_distribution[2])
                    right = cur_char
                    # img = np.append(, cur_char, axis=1)
                else:
                    left = img
                    right = cv2.copyMakeBorder(cur_char, top=int(abs_dif), left=int(space_size), right=0, bottom=0, borderType=cv2.BORDER_CONSTANT,
                                                       value=spacing_distribution[2])

                print(left.shape)
                print(right.shape)
                img = np.append(left, right, axis=1)
                print(img.shape)
                if verbose:
                    cv2.imshow("Line", img)
                    cv2.waitKey()

            #print(img.dtype)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_bilat = cv2.bilateralFilter(img_gray, -1, 35, 10)

            towrite = open(str(counter) + ".gt.txt", "w")
            towrite.write(text)
            towrite.close()
            cv2.imwrite(str(counter) + ".png", img_bilat)
            counter += 1

        #return imgs






    def extract_nonbin_symbols(self, pdfs, grayscale=False):
        for pdf, page_range in pdfs.items():
            bin_syms = []
            nonbin_syms = []
            text_name = path.splitext(pdf)[0].split('/')[-1]
            verbose = page_range[2]
            for page in self.pdf_to_jpg_stream(pdf, page_range[0], page_range[1]):
                try:
                    nonbin_page = self.resize_page(cv2.imread(page))
                    if grayscale:
                        nonbin_page = cv2.cvtColor(nonbin_page, cv2.COLOR_BGR2GRAY)

                    bin_page = self.transform_page(page, 9, 10)
                    x, y, w, h = self.get_text_region_bounds(bin_page) #upper right corner!
                    bin_crop = bin_page[y : (y + h), x : (x + w)]
                    nonbin_crop = nonbin_page[y : (y + h), x : (x + w)]

                    if verbose:
                        cv2.imshow("Binarized page", bin_page)
                        cv2.waitKey()
                        cv2.imshow("Original page", nonbin_page)
                        cv2.waitKey()
                        cv2.imshow("Original crop", nonbin_crop)
                        cv2.waitKey()
                        cv2.imshow("Binarized crop", bin_crop)
                        cv2.waitKey()

                    img, contours, hierarchy = self.get_contours(bin_crop)
                    for contour in contours:
                        c_x, c_y, c_w, c_h = cv2.boundingRect(contour)
                        if c_w > 25 or c_h > 25 or c_h < 5 or c_w < 5:
                            continue
                        sym_bin = bin_crop[c_y : (c_y + c_h), c_x : (c_x + c_w)]
                        sym_nonbin = nonbin_crop[c_y : (c_y + c_h), c_x : (c_x + c_w)]
                        if verbose:
                            cv2.imshow("Binarized symbol", sym_bin)
                            cv2.waitKey()
                            cv2.imshow("Original symbol", sym_nonbin)
                            cv2.waitKey()

                        bin_syms.append(sym_bin)
                        nonbin_syms.append(sym_nonbin)
                except AttributeError:
                    continue

            self.cluster_symbols(bin_syms, text_name, nonbin_syms=nonbin_syms)


    def extract_symbols(self, pdfs):
        #counter = 1
        #toreturn = []
        for pdf, page_range in pdfs.items():
            sym_array = []
            verbose = page_range[2]
            text_name = path.splitext(pdf)[0].split('/')[-1]
            for page in self.pdf_to_jpg_stream(pdf, page_range[0], page_range[1]):

                thr_image = self.transform_page(page, 9, 10)
                if verbose:
                    cv2.imshow(page + "_" + "threshold", thr_image)
                    cv2.waitKey()
                cropped_image = self.get_bounding_box(thr_image, verbose=verbose)
                if verbose:
                    cv2.imshow(page + "_" + "cropped", cropped_image)
                    cv2.waitKey()
                if cropped_image != None:
                    cr_im, contours, hierarchy = self.get_contours(cropped_image)
                    display_img = cr_im.copy()
                    cv2.drawContours(display_img, contours, -1, (255, 115, 10), 3)
                    if verbose:
                        cv2.imshow(page + "_" + "contours", display_img)
                        cv2.waitKey()
                    for image in self.cut_contours(cropped_image, contours):
                        sym_array.append(image)
            self.cluster_symbols(sym_array, text_name, verbose=verbose)
                        #cv2.imwrite("symbol_shapes/%d.png" % (counter), image)
                        #counter += 1
        #return toreturn


    def prepare_for_clustering(self, symbols, max_dims, verbose=False):
        #max_dims = (max([x.shape[0] for x in symbols]), max([y.shape[1] for y in symbols]))
        padded_symbols = []
        for symbol in symbols:
            if symbol != None and symbol.shape[0] <= max_dims[0] and symbol.shape[1] <= max_dims[1]:
                xdiff = max_dims[0] - symbol.shape[0]
                ydiff = max_dims[1] - symbol.shape[1]
                toappend = cv2.copyMakeBorder(symbol, bottom=xdiff, top=0, left=0, right=ydiff, borderType=cv2.BORDER_CONSTANT, value=0)
                #assert toappend.shape[:2] == (25, 25)
            else:
                toappend = np.zeros((max_dims[0], max_dims[1], 1), np.uint8)

            if verbose:
                cv2.imshow(repr(toappend.shape) , toappend)
                cv2.waitKey()

            padded_symbols.append([int(x != 0) for x in toappend.flatten()])

        return padded_symbols


    def cluster_symbols(self, symbols, directory, verbose=False, nonbin_syms=None):
        #paths = [y for y in listdir("symbol_shapes") if path.isfile("symbol_shapes/" + y)]
        #symbols = [cv2.imread("symbol_shapes/" + x, cv2.IMREAD_GRAYSCALE) for x in paths]
        selection = NMF(n_components=20, init="nndsvda")
        clustering = AffinityPropagation(damping=0.55, preference=-1)
        if not path.exists("symbol_shapes/" + directory):
            makedirs("symbol_shapes/" + directory)
        padded_symbols = self.prepare_for_clustering(symbols, (25, 25), verbose)
        symbol_traits = selection.fit_transform(padded_symbols)
        labels = clustering.fit_predict(symbol_traits)
        for i in range(len(labels)):
            curdir = "symbol_shapes/%s/%d"%(directory, labels[i])
            if not path.exists(curdir):
                makedirs(curdir)
            if not nonbin_syms:
                cv2.imwrite(curdir + "/" + str(i) + ".png", symbols[i])
            else:
                cv2.imwrite(curdir + "/" + str(i) + ".png", nonbin_syms[i])








if __name__ == "__main__":
    detection = LetterDetection(False, 4, 4, 15, 5, ntraits=100) #current optimal: 50, 30, 9, 10
    counter = 0

    detection.generate_phrases("kievan.pdf", "igorshost.txt")

    # #detection.cluster_symbols(symbols)








