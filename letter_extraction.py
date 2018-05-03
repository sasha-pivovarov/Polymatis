#encoding: utf-8
#from wand.image import Image, Color
#from PIL import Image

import cv2
import io
import numpy as np
import random
from datetime import datetime
from math import fabs
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, classification_report, silhouette_samples, accuracy_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA, NMF
from sklearn.pipeline import make_pipeline
import os
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GRU
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape, Input
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from os import listdir, makedirs, path
import time
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns


class LetterDetection:

    #detector = None
    #struct_element = None
    #sColor = None
    imagegen_aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1
                                  ,channel_shift_range=0, fill_mode="constant", cval=0, data_format="channels_first")
    imagegen_noaug = ImageDataGenerator(data_format="channels_first")
    #sSpace = None
    clahe = cv2.createCLAHE(2.0, (4, 4))
    special_chars = {"dot": "·"}
    placements = {"upper":[],
                  "lower":[]}
    #image_dump_path = None

    def __init__(self, areafilter, selement_h, selement_w, scolor, sspace, areamin=0, areamax=5000, log="log.txt"):
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
        self.logpath = log
        self.log('\n' + str(datetime.now()))

        #self.image_dump_path = image_path


    def log(self, string):
        with open(self.logpath, "a") as io:
            io.write(string + '\n')


    def transform_page(self, page, resize=True):
        """
        Convert a page to grayscale, apply the bilateral filter, threshold the image, in that order. Resizing optional.
        """
        image = cv2.imread(page, cv2.IMREAD_COLOR)
        if resize:
            image = resize_page(image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("Grayscale_img.png", image_gray)
        #image_gray = self.clahe.apply(image_gray)
        #image_gray = cv2.equalizeHist(image_gray)
        smoothed_image = cv2.bilateralFilter(image_gray, -1, self.sColor, self.sSpace)
        cv2.imwrite("Smoothed_img.png", smoothed_image)
        #thresholded_image = cv2.adaptiveThreshold(smoothed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
        ret, thresholded_image = cv2.threshold(smoothed_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        cv2.imwrite("Thresholded_img.png", thresholded_image)

        return thresholded_image


    def get_bounding_box(self, image, skewCorrection=False, verbose=False):
        """
        Returns the minimal rectangle around the text area. May also attemt to correct skew but the results are kinda gross so far.
        Deprecated, use get_text_region_bounds instead until I sort this out.
        """
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


    def get_text_region_bounds(self, bin_image):
        """
        Same as get_bounding_box, but just the essentials.
        """
        eroded_image = cv2.erode(bin_image, self.erosion_element)
        cv2.imwrite("Eroded_image.png", eroded_image)
        points = cv2.findNonZero(eroded_image)
        return cv2.boundingRect(points)


    def get_spacing_distribution(self, pdf, verbose, start=40, end=50):
        """
        Calculates the distribution of space sizes in the image. Also returns the average color.
        """
        crop = None
        answered = False
        avg_color = None

        while not answered:
            for page in pdf_to_jpg_stream(pdf, start, end):
                bin_page = self.transform_page(page)
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
                    avg_color = get_mean_color(page)
                    answered = True
                    break

        if verbose:
            swatch = np.array([[[avg_color] * 100] * 100], np.uint8)
            cv2.imshow("Average text color", swatch)
            cv2.waitKey()
        im, contours, hierarchy = get_contours(crop)
        boxes = [cv2.boundingRect(x) for x in contours]
        lines = []
        spaces = []
        for box in boxes:

            #box_xrange = box[0], box[0] + box[2]
            box_yrange = box[1], box[1] + box[3]
            if not [(box_yrange[0] in range(x[0], x[1]) or box_yrange[1] in range(x[0], x[1])) for x in [y[0] for y in lines]].__contains__(True):
                line_boxes = [linebox for linebox in boxes if linebox[1] in range(box_yrange[0], box_yrange[1]) or (linebox[1] + linebox[3]) in range(box_yrange[0], box_yrange[1])]
                lines.append((min([x[1] for x in line_boxes]), max((x[1] + x[3]) for x in line_boxes)))
                spaces.extend(boxes_to_spaces(line_boxes))

        return np.mean(spaces), np.std(spaces), avg_color


    def generate_phrases(self, pdf, text, verbose=False):
        """
        Generates the Unicode phrases in text as png images of lines, using the symbols in pdf.
        """
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
                    left = cv2.copyMakeBorder(img, top=0, left=0, right=int(space_size), bottom=int(abs_dif), borderType=cv2.BORDER_CONSTANT,
                                                       value=spacing_distribution[2])
                    right = cur_char
                    # img = np.append(, cur_char, axis=1)
                else:
                    left = img
                    right = cv2.copyMakeBorder(cur_char, top=0, left=int(space_size), right=0, bottom=int(abs_dif), borderType=cv2.BORDER_CONSTANT,
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

            img_bilat = cv2.bilateralFilter(img_gray, -1, 35, 8)

            towrite = open(str(counter) + ".gt.txt", "w")
            towrite.write(text)
            towrite.close()
            cv2.imwrite(str(counter) + ".png", img_bilat)
            counter += 1

        #return imgs


    def extract_nonbin_symbols(self, pdfs, grayscale=False):
        """
        Produces color images of the characters in pdfs.
        """
        for pdf, page_range in pdfs.items():
            bin_syms = []
            nonbin_syms = []
            text_name = path.splitext(pdf)[0].split('/')[-1]
            verbose = page_range[2]
            for page in pdf_to_jpg_stream(pdf, page_range[0], page_range[1]):
                try:
                    nonbin_page = resize_page(cv2.imread(page))
                    if grayscale:
                        nonbin_page = cv2.cvtColor(nonbin_page, cv2.COLOR_BGR2GRAY)

                    bin_page = self.transform_page(page)
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

                    img, contours, hierarchy = get_contours(bin_crop)
                    display_img = img.copy()
                    cv2.drawContours(display_img, contours, -1, (255, 115, 10), 3)
                    cv2.imwrite("Contours_img.png", display_img)

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

            yield bin_syms, nonbin_syms, text_name
            #self.cluster_symbols(bin_syms, text_name, nonbin_syms=nonbin_syms)


    def extract_and_cluster(self, pdfs):
        """
        Extracts symbols from the pdfs and then clusters them.
        """
        for syms in self.extract_nonbin_symbols(pdfs):
            cluster_symbols(syms[0], syms[2], nonbin_syms=syms[1])


    def extract_symbols(self, pdfs):
        """
        Produces binary images of the symbols in pdfs.
        """
        #counter = 1
        #toreturn = []
        for pdf, page_range in pdfs.items():
            sym_array = []
            verbose = page_range[2]
            text_name = path.splitext(pdf)[0].split('/')[-1]
            for page in pdf_to_jpg_stream(pdf, page_range[0], page_range[1]):

                thr_image = self.transform_page(page)
                if verbose:
                    cv2.imshow(page + "_" + "threshold", thr_image)
                    cv2.waitKey()
                cropped_image = self.get_bounding_box(thr_image, verbose=verbose)
                if verbose:
                    cv2.imshow(page + "_" + "cropped", cropped_image)
                    cv2.waitKey()
                if cropped_image != None:
                    cr_im, contours, hierarchy = get_contours(cropped_image)
                    display_img = cr_im.copy()
                    cv2.drawContours(display_img, contours, -1, (255, 115, 10), 3)
                    cv2.imwrite("Contours.png", display_img)
                    if verbose:
                        cv2.imshow(page + "_" + "contours", display_img)
                        cv2.waitKey()
                    for image in cut_contours(cropped_image, contours):
                        sym_array.append(image)
            cluster_symbols(sym_array, text_name, verbose=verbose)
                        #cv2.imwrite("symbol_shapes/%d.png" % (counter), image)
                        #counter += 1
        #return toreturn

    # reshaper = lambda x, y: tuple([1, x, y, 1])
    def prepare_for_classification(self, pdfnames, pad_out=True, flatten=True, padded_shape=(25, 25)):
        """
        Pads out the training symbols and makes label arrays for them.
        """
        symbols = []
        labels = []
        for name in pdfnames:
            foldername = "symbol_shapes/" + name
            dirnames = listdir(foldername)

            for dirname in dirnames:
                dirpath = foldername + "/" + dirname
                name = parse_name(dirname)
                syms = [self.transform_page(dirpath + "/" + x, False) for x in listdir(dirpath)]
                if pad_out and flatten:
                    new_symbols = prepare_for_clustering(syms, padded_shape)
                elif pad_out:
                    new_symbols = prepare_for_clustering(syms, padded_shape, flatten=False)
                else:
                    new_symbols = [x.reshape(tuple([1, x.shape[0], x.shape[1], 1])) for x in syms]
                new_labels = [name for x in range(len(syms))]
                assert len(new_labels) == len(new_symbols)
                symbols.extend(new_symbols)
                labels.extend(new_labels)


        return symbols, labels


    def classify_symbols(self, pdfs, training_folders):
        """
        Classifies the symbols in pdfs using a random forest trained using the symbols in training_folders.
        """
        training_syms, training_labels = self.prepare_for_classification(training_folders)
        for test_bins, test_nbs, textname in self.extract_nonbin_symbols(pdfs):
            test_padded = prepare_for_clustering(test_bins, (25, 25))
            classifier = GridSearchCV(RandomForestClassifier(), {"criterion":["gini", "entropy"], "max_features":["sqrt", "log2", None]})
            selector = NMF(100, "nndsvda")
            train = selector.fit_transform(training_syms)
            test = selector.transform(test_padded)
            classifier.fit(train, training_labels)
            predicted = classifier.predict(test)

            foldername = "symbol_shapes/"+ textname + "/"
            for i in range(len(predicted)):
                curdir = foldername + predicted[i] + "_pred/"
                if not path.exists(curdir):
                    makedirs(curdir)
                cv2.imwrite(curdir + str(i) + ".png", test_nbs[i])


    def classifier_tests(self, symbols, labels):
        textfile = open("classification_reports_"  + str(time.time()) + ".txt", "a")
        selection = NMF(n_components=100, init="nndsvda")
        traits = selection.fit_transform(symbols, labels)
        X_train, X_test, y_train, y_test = train_test_split(traits, labels, test_size=0.33)

        bayes = GridSearchCV(BernoulliNB(), {"alpha":[0, 0.5, 1.0, 5, 10]})
        #logit = GridSearchCV(LogisticRegression(), {"multi_class":["ovr"], "C":[0.5, 1.0, 5.0]})
        forest = GridSearchCV(RandomForestClassifier(), {"criterion":["gini", "entropy"], "max_features":["sqrt", "log2", None]})
        #adaboost = GridSearchCV(AdaBoostClassifier(), {"base_estimator":[BernoulliNB(alpha=0.5), None],"n_estimators":[50, 100], "learning_rate":[0.5, 1.0, 5.0]})
        gradient = GridSearchCV(GradientBoostingClassifier(), {"learning_rate":[0.05, 0.1, 0.5], "n_estimators":[100, 150]})

        for classifier in bayes, forest, gradient:
            classifier.fit(X_train, y_train)
            textfile.write("\n=============================\n")
            textfile.write(repr(classifier.best_params_))
            pred = classifier.predict(X_test)
            textfile.write(classification_report(y_test, pred))

        textfile.close()

    def create_nn(self, optimizer="sgd") -> Sequential:
        model = Sequential()

        model.add(Conv2D(filters=32, input_shape=(1, 24, 24), kernel_size=(3, 3), data_format="channels_first"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

        model.add(Conv2D(64, (3, 3), data_format="channels_first"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

        model.add(Conv2D(64, (3, 3), data_format="channels_first"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        #model.add(Reshape(target_shape=(4, 4*16)))
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dropout(0.5))
        #model.add(GRU(64, return_sequences=True, kernel_initializer="he_normal"))
        #model.add(GRU(64, return_sequences=True, go_backwards=True, kernel_initializer="he_normal"))
        #model.add(GRU(64, return_sequences=True, kernel_initializer="he_normal"))
        #model.add(GRU(64, return_sequences=True, go_backwards=True, kernel_initializer="he_normal"))
        model.add(Dense(39, activation="softmax"))
        model.summary()

        model.summary(print_fn=self.log)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return model


    def nn_tests(self, symbols, labels, optimizer, save_path, encoder_path="label_encoder.pkl"):
        # i = 0
        # for batch in self.imagegen.flow(symbols[0], batch_size=1,
        #                           save_to_dir='preview', save_prefix='sym', save_format='png'):
        #     i += 1
        #     if i > 20:
        #         break
        self.log("\n++++++++++++++++++++++++++++++++++\n" + optimizer + "\n")
        print(len(set(labels)))
        print(len(symbols))
        acc = 0
        prec = 0
        X = np.array(symbols)
        y = np.array(labels)
        f1 = 0
        skf = StratifiedKFold(n_splits=4)
        encoder = load_encoder("models/" + encoder_path)
        for train_index, test_index in skf.split(X, y):
            sym_train, sym_test = X[train_index], X[test_index]
            lab_train, lab_test = y[train_index], y[test_index]


            n_augmented = 0
            aug_labels = []
            aug_symbols = []
            lab_train = to_categorical(encoder.transform(lab_train))
            lab_test = to_categorical(encoder.transform(lab_test))
            data_noaug = self.imagegen_noaug.flow(sym_train, lab_train)
            data_aug = self.imagegen_aug.flow(sym_train, lab_train)

            model1 = self.create_nn(optimizer=optimizer)

            model1.fit_generator(data_noaug, steps_per_epoch=500, epochs=50)
            #model1.load_weights("model1.h5")
            model1.save_weights("models/" + save_path)
            self.log(f"Saved weights as {save_path}")

            # model2 = self.create_nn()
            # model2.fit_generator(data_aug, steps_per_epoch=500, epochs=50)
            # # model2.load_weights("model2.h5")
            # model2.save_weights("model2_2layers.h5")

            pred1 = convert_nn_output(model1.predict(sym_test))
            # pred2 = convert_nn_output(model2.predict(np.array(sym_test)))
            acc += accuracy_score(lab_test, pred1)
            prec += precision_score(lab_test, pred1, average="samples")
            f1 += f1_score(lab_test, pred1, average="samples")
            print(classification_report(encoder.inverse_transform(lab_test), encoder.inverse_transform(pred1)))
            self.log(classification_report(encoder.inverse_transform(lab_test), encoder.inverse_transform(pred1)) + '\n======================================\n')


        self.log(
            """
        ACCURACY: %s
        PRECISION %s
        F1: %s
        """ % (str(acc/4.0), str(prec / 4.0), str(f1 / 4.0)))
        print("++++++++++++++++++++++++++++++++++++++++++++++")
        # print(classification_report(lab_test, pred2))


    def classifier_tests_secondstage(self, X_tr, y_tr, X_test, y_test):
        textfile = open("classification_reports_"  + str(time.time()) + "sstage.txt", "a")
        selection = NMF(n_components=100, init="nndsvda")
        traits = selection.fit_transform(symbols, labels)
        #X_train, X_test, y_train, y_test = train_test_split(traits, labels, test_size=0.33)

        bayes = GridSearchCV(BernoulliNB(), {"alpha":[0, 0.5, 1.0, 5, 10]})
        #logit = GridSearchCV(LogisticRegression(), {"multi_class":["ovr"], "C":[0.5, 1.0, 5.0]})
        forest = GridSearchCV(RandomForestClassifier(), {"criterion":["gini", "entropy"], "max_features":["sqrt", "log2", None]})
        #adaboost = GridSearchCV(AdaBoostClassifier(), {"base_estimator":[BernoulliNB(alpha=0.5), None],"n_estimators":[50, 100], "learning_rate":[0.5, 1.0, 5.0]})
        gradient = GridSearchCV(GradientBoostingClassifier(), {"learning_rate":[0.05, 0.1, 0.5], "n_estimators":[100, 150]})

        for classifier in bayes, forest, gradient:
            classifier.fit(X_tr, y_tr)
            textfile.write("\n=============================\n")
            textfile.write(repr(classifier.best_params_))
            pred = classifier.predict(X_test)
            textfile.write(classification_report(y_test, pred))

        textfile.close()


    def clustering_tests(self, pdfs):
        for syms in self.extract_nonbin_symbols(pdfs):
            print("%d symbols"%len(syms[0]))
            padded_bins = prepare_for_clustering(syms[0], (25, 25))
            selection = {"nmf50": NMF(50, init="nndsvda"),
                         "nmf100": NMF(100, init="nndsvda"),
                         "nmf300": NMF(300, init="nndsvda"),
                         "nmf200nonnndsvda": NMF(200),
                         "nmf200": NMF(200, init="nndsvda"),
                         "pca50": PCA(50),
                         "pca100": PCA(100),
                         "pca200": PCA(200)
                         }
            clustering = {
                          "AP05":AffinityPropagation(.5),
                          "AP08":AffinityPropagation(.8)
                          ,
                          "DB05":DBSCAN(),
                          "DB08":DBSCAN(.8),
                          "MS": MeanShift()
                         }

            for sel_name, selector in selection.items():
                for cl_name, clusterer in clustering.items():
                    print(sel_name + "_" + cl_name + ":\n")
                    selected = selector.fit_transform(padded_bins)
                    clustered = clusterer.fit_predict(selected)
                    print(silhouette_score(selected, clustered))
                    print(len(set(clustered)))

    def test_segsystem_on_page(self, page, model_path, encoder_path):
        img = cv2.imread(page)
        img = self.transform_page(img)
        img = self.get_bounding_box(img)
        im, contours, hierarchy = get_contours(img)
        lines = sort_contours(contours)
        line_crops = [cut_contours_new(img, x) for x in lines]
        model = Sequential()
        model.load_weights("models/" + model_path)
        encoder = load_encoder(encoder_path)
        line_preds = [model.predict(line) for line in line_crops]
        text = "\n".join("".join(encoder.inverse_transform(line_preds)))
        with open("test.txt", "w") as io:
            io.write(text)


    def get_symbol_size_counts(self, pages):
        contours_all = []
        for page in pages:
            img = cv2.imread(page)
            img = self.transform_page(img)
            img = self.get_bounding_box(img)
            im, contours, hierarchy = get_contours(img)
            contours_all.extend(contours)
        areas = [cv2.contourArea(x) for x in contours_all]
        mean_area = np.mean(areas)
        sd_area = np.std(areas)
        median_area = np.median(areas)
        print(f"MEAN: {mean_area}, MEDIAN: {median_area}, DEVIATION: {sd_area}")
        sns.distplot(areas)
        plt.show()












if __name__ == "__main__":
    detection = LetterDetection(False, 4, 4, 15, 5) #current optimal: 50, 30, 9, 10
    #counter = 0
    #
    # detection.extract_and_cluster({"kievgospel.pdf":(70, 90, False),
    #                                   "ostromirovo.pdf":(70, 90, False),
    #                                    "kievan.pdf":(70, 90, False)})
    # detection.clustering_tests({"kievgospel.pdf":(30, 40, False)})
    symbols, labels = detection.prepare_for_classification(["kievgospel", "ostromirovo", "kievan"], pad_out=True, flatten=False, padded_shape=(24, 24))
    # sym_test, label_test = detection.prepare_for_classification(["ostromirovo"])
    # detection.classify_symbols({"kievgospel.pdf": (20, 40, False)}, ["kievgospel"])
    # #detection.cluster_symbols(symbols)
    #detection.classifier_tests_secondstage(symbols, labels, sym_test, label_test)
    save_encoder(labels, "label_encoder.pkl")
    detection.nn_tests(symbols, labels, "sgd", "model3_3layers_sgd.h5")
    detection.nn_tests(symbols, labels, "adam", "model3_3layers_adam.h5")
    detection.nn_tests(symbols, labels, "rmsprop", "model3_3layers_rmsprop.h5")








