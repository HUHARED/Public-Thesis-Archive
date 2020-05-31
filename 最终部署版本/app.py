from flask import Flask, jsonify, request, render_template, url_for
from flask_cors import CORS
from . import handleImage

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return render_template("./index.html")
    # 不知道为什么 return render_template(url_for("index")) 报错，之后再说吧。


@app.route("/cats_classify/", methods=["POST"])
def cats_classify():

    imgStream = request.files.get("file").read()

    """ 判断猫数量 """
    catNumber = handleImage.check_cat_number(imgStream)
    if catNumber > 1 or catNumber == 0:
        if catNumber > 1:
            finalMsg = "<strong>图片里的猫好像不只一只哦</strong>"
        if catNumber == 0:
            finalMsg = "<strong>图片里好像没有猫哦</strong>"
        res = {
            'data': {
                'msg': finalMsg,
            }
        }
        return jsonify(res)

    """ 如果猫数量为1，进行下面步骤 """

    predInfoList = handleImage.check_cat_class(imgStream)
    finalMsg = ""
    for ele in predInfoList:
        [breedEn, breedCh, encyclopediaPet, encyclopediaWiki,
            encyclopediaBaidu, confiLivel] = ele

        msgBreedEn = '<br><br><strong style="font-size: 36px;">{}</strong><br/>'.format(
            breedEn)
        msgBreedCh = '<span style="font-size: 24px;">(<strong>{}</strong>)</span><br/>'.format(
            breedCh)
        msgConfiLivel = '<span style="font-size: 16px;">可信度：<strong>{}</strong></span><br/>'.format(
            confiLivel)

        baseStr = '<a target="_blank" href="{url}">{name}</a><span class="el-icon-view el-icon--right"></span><br/>'
        if type(encyclopediaPet) != str:
            msgEncyclopediaPet = ''
        else:
            msgEncyclopediaPet = baseStr.format(
                url=encyclopediaPet, name="宠物百科")
        if type(encyclopediaWiki) != str:
            msgEncyclopediaWiki = ''
        else:
            msgEncyclopediaWiki = baseStr.format(
                url=encyclopediaWiki, name="维基百科")
        if type(encyclopediaBaidu) != str:
            msgEncyclopediaBaidu = ''
        else:
            msgEncyclopediaBaidu = baseStr.format(
                url=encyclopediaBaidu, name="百度百科")

        thisBreedMsg = msgBreedEn+msgBreedCh+msgConfiLivel+msgEncyclopediaPet + \
            msgEncyclopediaWiki + msgEncyclopediaBaidu

        finalMsg += thisBreedMsg + "<br/><br/>"

    res = {
        'data': {
            'msg': finalMsg,
        }
    }
    # 返回json数据
    return jsonify(res)
