package com.intel.analytics.zoo.examples.nnframes.xgboost

import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.nnframes.{NNFileReader, XGBClassifierModel}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{SQLContext, SparkSession}

object XgboostExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    val df = spark.read.format("csv")
      .option("sep", ",")
      .option("inferSchema", true)
      .option("header", true)
      .load(args(1))
//    val df = NNFileReader.readCSV(args(1), spark.sparkContext)
    val path = args(0)
    val yuyinPath = "xgb_yuyin-18-16.model"
    val model2 = XGBClassifierModel.load(path + yuyinPath, 3)
    model2.setFeaturesCol(Array("分公司名称", "用户入网时间", "用户状态", "年龄", "性别", "用户星级",
      "是否集团成员", "城市农村用户", "是否欠费用户", "主套餐费用","通话费用", "通话费用趋势","VoLTE掉话率", "ESRVCC切换时延",
      "ESRVCC切换比例", "ESRVCC切换成功率", "VoLTE接续时长", "呼叫建立时长", "VoLTE接通率",
      "全程呼叫成功率", "VoLTE掉话率_diff",
      "ESRVCC切换时延_diff", "ESRVCC切换比例_diff", "ESRVCC切换成功率_diff",
      "VoLTE接续时长_diff", "呼叫建立时长_diff", "VoLTE接通率_diff", "全程呼叫成功率_diff"))
    model2.setPredictionCol("yuyin")
    val yuyin = model2.transform(df).select("yuyin", "手机号码")

    val shoujiPath = "xgb_shouji-18-16.model"
    val shouji_model = XGBClassifierModel.load(path + shoujiPath, 3)
    shouji_model.setFeaturesCol(Array("分公司名称", "用户入网时间", "用户状态", "年龄", "性别", "用户星级",
      "是否集团成员", "城市农村用户", "主套餐费用", "流量费用", "流量费用趋势",
      "网页响应成功率", "网页响应时延", "网页显示成功率", "网页浏览成功率", "网页打开时长",
      "视频响应成功率", "视频响应时延", "视频平均每次播放卡顿次数", "视频播放成功率", "视频播放等待时长",
      "即时通信接入成功率", "即时通信接入时延", "下载速率", "上传速率",
      "网页响应成功率_diff", "网页响应时延_diff", "网页显示成功率_diff", "网页浏览成功率_diff",
      "网页打开时长_diff", "视频响应成功率_diff", "视频响应时延_diff", "视频平均每次播放卡顿次数_diff",
      "视频播放成功率_diff", "视频播放等待时长_diff",
      "即时通信接入成功率_diff", "即时通信接入时延_diff", "下载速率_diff", "上传速率_diff"))
    shouji_model.setPredictionCol("shouji")
    val shouji = shouji_model.transform(df).select("shouji", "手机号码")

    val zifeiDF=df.join(shouji, "手机号码").join(yuyin, "手机号码")
    val zifeiPath = "xgb_zifei-18-16.model"
    val zifei_model = XGBClassifierModel.load(path + zifeiPath, 3)
    zifei_model.setFeaturesCol(Array("年龄", "性别", "用户入网时间", "用户星级", "是否集团成员", "城市农村用户", "是否欠费用户",
      "主套餐费用", "超套费用", "通话费用", "通话费用趋势", "流量费用", "流量费用趋势",
      "近3月的平均出账费用", "近3月的平均出账费用趋势", "近3月超套平均", "近3月月均欠费金额","用户状态","分公司名称",
      "yuyin", "shouji"))
    zifei_model.setPredictionCol("zifei")
    val zifei = zifei_model.transform(zifeiDF).select("zifei", "shouji", "yuyin", "手机号码")

    val manyiDF=df.join(zifei, "手机号码")
    val manyiPath = "xgb_manyi-18-16.model"
    val manyi_model = XGBClassifierModel.load(path + zifeiPath, 3)
    manyi_model.setFeaturesCol(Array("年龄", "性别", "用户入网时间", "用户星级", "zifei",
      "yuyin", "shouji"))
    manyi_model.setPredictionCol("manyi")
    val manyi = manyi_model.transform(manyiDF)
    manyi.show(1)
  }

}
