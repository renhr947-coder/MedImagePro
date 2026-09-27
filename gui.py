import cv2
import numpy as np
import os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


from core.image_reader import read_image

from core.enhancement import enhance_image

from core.denoise import gaussian_denoise

from core.segmentation import (
    threshold_segmentation,
    edge_segmentation
)

from core.feature import (
    calculate_features,
    save_report
)

from utils.visualization import create_overlay



class MedicalImageApp(QWidget):


    def __init__(self):

        super().__init__()

        self.image_path = None

        self.initUI()

    def initUI(self):

        self.setWindowTitle(
            "医学图像处理软件"
        )

        self.resize(
            1200,
            850
        )

        main_layout = QVBoxLayout()

        # =========================
        # 顶部标题
        # =========================

        title = QLabel(
            "智能医学图像处理平台"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        title.setStyleSheet(
            """
            QLabel{

                font-size:26px;

                font-weight:bold;

                color:#1F4E79;

                padding:15px;

            }
            """
        )

        main_layout.addWidget(title)

        subtitle = QLabel(
            "MRI Enhancement | Segmentation | AI Analysis"
        )

        subtitle.setAlignment(
            Qt.AlignCenter
        )

        subtitle.setStyleSheet(
            """
            QLabel{

                color:#666;

                font-size:13px;

                padding-bottom:15px;

            }
            """
        )

        main_layout.addWidget(subtitle)

        # =========================
        # 图像显示区域
        # =========================

        image_layout = QHBoxLayout()

        self.original_viewer = QLabel(
            "原始医学图像"
        )

        self.enhanced_viewer = QLabel(
            "图像预处理"
        )

        self.segment_viewer = QLabel(
            "AI分割结果"
        )

        for viewer in [

            self.original_viewer,

            self.enhanced_viewer,

            self.segment_viewer

        ]:
            viewer.setAlignment(
                Qt.AlignCenter
            )

            viewer.setFixedSize(
                320,
                320
            )

            viewer.setStyleSheet(
                """
                QLabel{

                    background:white;

                    border:2px solid #D9E6F2;

                    border-radius:12px;

                    color:#555;

                    font-size:15px;

                }
                """
            )

            image_layout.addWidget(
                viewer
            )

        main_layout.addLayout(
            image_layout
        )

        # =========================
        # 图像预处理
        # =========================

        label1 = QLabel(
            "图像预处理"
        )

        label1.setStyleSheet(
            """
            QLabel{

                color:#1F4E79;

                font-size:16px;

                font-weight:bold;

                padding-top:10px;

            }
            """
        )

        main_layout.addWidget(label1)

        self.open_btn = QPushButton(
            "打开医学图像"
        )

        self.open_btn.clicked.connect(
            self.open_image
        )

        self.open_btn.setStyleSheet(
            """
            QPushButton{

                background:#5B8FF9;

                color:white;

                border-radius:8px;

                height:38px;

                font-size:14px;

            }

            QPushButton:hover{

                background:#3D73D5;

            }
            """
        )

        self.enhance_btn = QPushButton(
            "图像增强"
        )

        self.enhance_btn.clicked.connect(
            self.enhance
        )

        self.enhance_btn.setStyleSheet(
            """
            QPushButton{

                background:#2F80ED;

                color:white;

                border-radius:8px;

                height:38px;

                font-size:14px;

            }

            QPushButton:hover{

                background:#1C64C8;

            }
            """
        )

        self.denoise_btn = QPushButton(
            "图像去噪"
        )

        self.denoise_btn.clicked.connect(
            self.denoise
        )

        self.denoise_btn.setStyleSheet(
            """
            QPushButton{

                background:#00A6A6;

                color:white;

                border-radius:8px;

                height:38px;

                font-size:14px;

            }

            QPushButton:hover{

                background:#008080;

            }
            """
        )

        main_layout.addWidget(
            self.open_btn
        )

        main_layout.addWidget(
            self.enhance_btn
        )

        main_layout.addWidget(
            self.denoise_btn
        )

        # =========================
        # 智能分析
        # =========================

        label2 = QLabel(
            "智能分析"
        )

        label2.setStyleSheet(
            """
            QLabel{

                color:#1F4E79;

                font-size:16px;

                font-weight:bold;

                padding-top:10px;

            }
            """
        )

        main_layout.addWidget(label2)

        self.segment_btn = QPushButton(
            "传统图像分割"
        )

        self.segment_btn.clicked.connect(
            self.segment
        )

        self.segment_btn.setStyleSheet(
            """
            QPushButton{

                background:#4472C4;

                color:white;

                border-radius:8px;

                height:38px;

                font-size:14px;

            }
            """
        )

        self.ai_btn = QPushButton(
            "AI智能分割 (U-Net)"
        )

        self.ai_btn.clicked.connect(
            self.ai_segmentation
        )

        self.ai_btn.setStyleSheet(
            """
            QPushButton{

                background:#8E44AD;

                color:white;

                border-radius:8px;

                height:40px;

                font-size:14px;

                font-weight:bold;

            }


            QPushButton:hover{

                background:#732D91;

            }

            """
        )

        main_layout.addWidget(
            self.segment_btn
        )

        main_layout.addWidget(
            self.ai_btn
        )

        # =========================
        # 定量分析
        # =========================

        label3 = QLabel(
            "定量分析"
        )

        label3.setStyleSheet(
            """
            QLabel{

                color:#1F4E79;

                font-size:16px;

                font-weight:bold;

                padding-top:10px;

            }
            """
        )

        main_layout.addWidget(label3)

        self.feature_btn = QPushButton(
            "特征分析"
        )

        self.feature_btn.clicked.connect(
            self.feature_analysis
        )

        self.feature_btn.setStyleSheet(
            """
            QPushButton{

                background:#27AE60;

                color:white;

                border-radius:8px;

                height:38px;

                font-size:14px;

            }


            QPushButton:hover{

                background:#1E8449;

            }

            """
        )

        main_layout.addWidget(
            self.feature_btn
        )

        # =========================
        # 全局背景
        # =========================

        self.setStyleSheet(
            """
            QWidget{

                background:#F5F8FC;

            }
            """
        )

        self.setLayout(
            main_layout
        )



    # =========================
    # 显示图片函数
    # =========================

    def show_image(
            self,
            path,
            viewer
    ):


        pixmap = QPixmap(
            path
        )


        viewer.setPixmap(

            pixmap.scaled(

                280,
                280,
                Qt.KeepAspectRatio

            )

        )



    # =========================
    # 打开图像
    # =========================

    def open_image(self):


        filename,_ = QFileDialog.getOpenFileName(

            self,

            "选择医学图像",

            "",

            "Images (*.png *.jpg *.bmp *.tif)"

        )



        if filename:


            self.image_path = filename


            self.show_image(

                filename,

                self.original_viewer

            )



    # =========================
    # 图像增强
    # =========================

    def enhance(self):


        if self.image_path:


            img = read_image(

                self.image_path

            )


            result = enhance_image(
                img
            )


            os.makedirs(
                "results",
                exist_ok=True
            )


            cv2.imwrite(

                "results/enhanced.png",

                result

            )


            self.show_image(

                "results/enhanced.png",

                self.enhanced_viewer

            )



    # =========================
    # 去噪
    # =========================

    def denoise(self):


        if self.image_path:


            img = read_image(
                self.image_path
            )


            result = gaussian_denoise(
                img
            )


            os.makedirs(
                "results",
                exist_ok=True
            )


            cv2.imwrite(

                "results/denoise.png",

                result

            )


            self.show_image(

                "results/denoise.png",

                self.enhanced_viewer

            )



    # =========================
    # 传统分割
    # =========================

    def segment(self):


        if self.image_path:


            choice = QMessageBox.question(

                self,

                "选择方法",

                "Yes:Canny边缘检测\nNo:Otsu阈值分割"

            )


            img = read_image(

                self.image_path

            )


            if choice == QMessageBox.Yes:


                result=edge_segmentation(
                    img
                )


            else:


                result=threshold_segmentation(
                    img
                )



            os.makedirs(
                "results",
                exist_ok=True
            )


            cv2.imwrite(

                "results/segmentation.png",

                result

            )



            overlay=create_overlay(

                img,

                result

            )


            cv2.imwrite(

                "results/overlay.png",

                overlay

            )


            self.show_image(

                "results/overlay.png",

                self.segment_viewer

            )



    # =========================
    # AI U-Net分割
    # =========================

    def ai_segmentation(self):

        from ai.predict import ai_predict

        from utils.metrics import (
            calculate_dice,
            calculate_iou
        )

        if self.image_path is None:
            return

        mask = ai_predict(
            self.image_path
        )

        image = read_image(
            self.image_path
        )

        overlay = create_overlay(
            image,
            mask
        )

        cv2.imwrite(
            "results/ai_overlay.png",
            overlay
        )

        self.show_image(
            "results/ai_overlay.png",
            self.segment_viewer
        )

        # ===================
        # AI报告
        # ===================

        area = np.sum(mask > 0)
        if area == 0:

            lesion_status = "No lesion detected"

        else:

            lesion_status = "Lesion detected"

        ratio = area / (mask.shape[0] * mask.shape[1])

        text = f"""

    AI分割分析报告


    Image Size:
    {mask.shape[0]} × {mask.shape[1]}


    Tumor Pixels:
    {area}


    Region Ratio:
    {ratio * 100:.2f}%


    Dice:
    需要真实mask验证


    IoU:
    需要真实mask验证

    """

        QMessageBox.information(

            self,

            "AI医学图像分析",

            text

        )


    # =========================
    # 特征分析
    # =========================

    def feature_analysis(self):


        if self.image_path:


            image=read_image(

                self.image_path

            )


            mask=cv2.imread(

                "results/segmentation.png",

                0

            )



            if mask is None:


                QMessageBox.warning(

                    self,

                    "提示",

                    "请先进行分割"

                )

                return



            features=calculate_features(

                image,

                mask

            )


            save_report(
                features
            )



            text=""


            for k,v in features.items():

                text += (

                    k

                    +

                    ": "

                    +

                    v

                    +

                    "\n"

                )


            QMessageBox.information(

                self,

                "医学图像分析报告",

                text

            )