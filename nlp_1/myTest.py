# coding= utf-8
# from keras_albert_model import build_albert

def reverse_side(s):
    rtn_s = ""
    for index, item in enumerate(s):
        if item == "左":
            rtn_s += "右"
        elif item == "右":
            rtn_s += "左"
        else:
            rtn_s += item
    return rtn_s


def reverse_sentence(sentence, label):
    rtn_label = label
    for index, item in enumerate(rtn_label):
        rtn_label[index] = reverse_side(item)
    rtn_sentence = reverse_side(sentence)
    return rtn_sentence, label


if __name__ == "__main__":
    # rtn = reverse_side("左侧额、颞叶见片状低密度影，左侧额、颞部颅板下方见线形稍高密度影，邻近见引流管影", ["左额叶", "左颞叶"])
    # rtn = reverse_side("右侧额、枕叶白质、左侧半卵圆中心及侧脑室旁白质可见散在片状密度减低影，部分边界清，邻近见引流管影", ["右额叶", "右枕叶","左半卵圆中心"])
    # print(rtn)
    # print(time.time().to)

    # model = build_albert(token_num=30000, training=True)
    # model.summary()

    import tensorflow as tf

    ones = tf.Variable(tf.ones([3, 3]))

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(ones))
