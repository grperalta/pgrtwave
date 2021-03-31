"""
Module for output string coloring and highlighting. Function name represents
the color of the text. For bold text use the suffix '_b'.
"""

__author__ = "Gilbert Peralta"

def purple(string_text):
    return "\033[035m" + string_text + "\033[0m"

def yellow(string_text):
    return "\033[033m" + string_text + "\033[0m";

def green(string_text):
    return "\033[032m" + string_text + "\033[0m";

def blue(string_text):
    return "\033[034m" + string_text + "\033[0m";

def cyan(string_text):
    return "\033[036m" + string_text + "\033[0m";

def red(string_text):
    return "\033[031m" + string_text+ "\033[0m";

def purple_b(string_text):
    return "\033[1m\033[035m" + string_text + "\033[0m";

def yellow_b(string_text):
    return "\033[1m\033[033m" + string_text + "\033[0m";

def green_b(string_text):
    return "\033[1m\033[032m" + string_text + "\033[0m";

def blue_b(string_text):
    return "\033[1m\033[034m" + string_text + "\033[0m";

def cyan_b(string_text):
    return "\033[1m\033[036m" + string_text + "\033[0m";

def red_b(string_text):
    return "\033[1m\033[031m" + string_text  + "\033[0m";

def highlight(string_text):
    return "\033[100m" + string_text  + "\033[49m";
