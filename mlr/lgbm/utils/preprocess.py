"""
CSVに変換する
"""
import csv
import re


def convert_text_to_csv(text_file_path, output_csv_path, column_names):
    """
    SVMLight/LibSVM形式のテキストデータをCSVに変換する
    """