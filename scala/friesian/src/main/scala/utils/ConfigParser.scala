package utils

import com.fasterxml.jackson.databind.{DeserializationFeature, MapperFeature, ObjectMapper, SerializationFeature}
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory

class ConfigParser(configPath: String) {
  def loadConfig(): gRPCHelper = {
    try {
      val configStr = scala.io.Source.fromFile(configPath).mkString
      val mapper = new ObjectMapper(new YAMLFactory())
      mapper.configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true)
      mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
      mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false)
      val helper = mapper.readValue[gRPCHelper](configStr, classOf[gRPCHelper])
      helper.parseConfigStrings()
      helper
    }
    catch {
      case e: Exception =>
        println(s"Invalid configuration, please check type regulations in config file")
        e.printStackTrace()
        throw new Error("Configuration parsing error")
    }
  }

}
