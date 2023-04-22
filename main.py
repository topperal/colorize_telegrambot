import telebot
from telebot import types
import os
import time

import argparse
import matplotlib.pyplot as plt
from colorization.colorizers import *

bot = telebot.TeleBot('6035021428:AAGhB49BPKl4e7QWyx6ogoYCTzNVS0aNR5Y')

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Пришлите фотографию, которую хотите раскрасить')

@bot.message_handler(content_types=['photo'])
def get_photo(message):
    markup = types.InlineKeyboardMarkup()
    btn1 = types.InlineKeyboardButton('Узнать больше о модели', url='http://richzhang.github.io/colorization/')
    markup.row(btn1)
    #btn2 = types.InlineKeyboardButton('Удалить фото', callback_data='delete')
    #btn3 = types.InlineKeyboardButton('Раскрасить фото', callback_data='colorize')
    #markup.row(btn2)
    bot.reply_to(message, 'Подождите немного, фотография раскрашивается...', reply_markup=markup)

    max_length = len(message.photo)
    raw = message.photo[max_length-1].file_id
    name = raw + ".jpg"
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(name, 'wb') as new_file:
        new_file.write(downloaded_file)
    colorize(name, message)

# @bot.callback_query_handler(func=lambda callback:True)
# def callback_message(callback):
#     if callback.data == 'delete':
#         bot.delete_message(callback.message.chat.id, callback.message.message_id-1)
#     elif callback.data == 'edit':
#         bot.edit_message_text('edit text', callback.message.chat.id, callback.message.message_id)

def colorize(name, message):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, default=name)
    parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
    parser.add_argument('-o', '--save_prefix', type=str, default='saved',
                        help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
    opt = parser.parse_args()

    # load colorizers
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    if (opt.use_gpu):
        colorizer_eccv16.cuda()
        colorizer_siggraph17.cuda()

    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img(opt.img_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    if (opt.use_gpu):
        tens_l_rs = tens_l_rs.cuda()

    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    plt.imsave('%s_eccv16.png' % opt.save_prefix, out_img_eccv16)
    plt.imsave('%s_siggraph17.png' % opt.save_prefix, out_img_siggraph17)

    photo16 = open('saved_eccv16.png', 'rb')
    bot.send_photo(message.chat.id, photo16, caption="Модель от 2016 года показывает такой результат")

    photo17 = open('saved_siggraph17.png', 'rb')
    bot.send_photo(message.chat.id, photo17, caption="Модель от 2017 года показывает такой результат")

    time.sleep(5)
    if os.path.exists(name):
        os.remove(name)
        print(f"File '{name}' has been deleted.")

bot.polling(none_stop=True)
