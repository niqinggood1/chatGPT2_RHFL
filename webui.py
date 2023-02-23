#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dialogGPT2_RHFL 
@File    ：webui.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2023/2/23 14:36 
'''
import gradio as gr
import cv2
# def to_black(image):
#     output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return output
# interface = gr.Interface(fn=to_black, inputs="image", outputs="image")
# interface.launch()


global i
i=0

import gradio as gr
def predict(input, history=[]):
    global i

    history.append( input+str(21) )
    i = i + 1
    response = i
    return response, history

gr.Interface(fn=predict,
             inputs=["text", "state"],
             outputs=["text", "state"]).launch()

if __name__ == '__main__':
    exit()
    
  
  