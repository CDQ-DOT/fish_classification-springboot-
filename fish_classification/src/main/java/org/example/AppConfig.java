package org.example;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.ClientHttpRequestFactory;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.http.converter.FormHttpMessageConverter;
import org.springframework.http.converter.StringHttpMessageConverter;
import org.springframework.web.client.RestTemplate;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

@Configuration // 必须加这个注解，标识这是配置类
public class AppConfig {

    // 配置 RestTemplate 的 Bean
    @Bean
    public RestTemplate restTemplate(ClientHttpRequestFactory factory) {
        RestTemplate restTemplate = new RestTemplate(factory);

        // 配置消息转换器（解决文件上传、中文乱码）
        List<org.springframework.http.converter.HttpMessageConverter<?>> converters = new ArrayList<>();
        converters.add(new StringHttpMessageConverter(StandardCharsets.UTF_8));
        converters.add(new FormHttpMessageConverter());
        restTemplate.setMessageConverters(converters);

        return restTemplate;
    }

    // 配置请求超时
    @Bean
    public ClientHttpRequestFactory simpleClientHttpRequestFactory() {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setReadTimeout(30000);  // 读取超时30秒
        factory.setConnectTimeout(10000); // 连接超时10秒
        return factory;
    }

    // 若 AppConfig 原有其他配置，继续保留即可
}