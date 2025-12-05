package org.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

@SpringBootApplication
// 显式扫描controller包（解决包扫描遗漏问题）
@ComponentScan(basePackages = {"org.example", "org.example.controller"})
public class FishClassificationApplication {
    public static void main(String[] args) {
        SpringApplication.run(FishClassificationApplication.class, args);
    }
}