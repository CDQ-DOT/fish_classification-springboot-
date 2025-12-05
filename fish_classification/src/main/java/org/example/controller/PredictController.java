package org.example.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.core.io.ByteArrayResource;

import java.io.IOException;

@RestController
public class PredictController {
    @Autowired
    private RestTemplate restTemplate;

    // Python FastAPI服务地址
    private static final String MODEL_API_URL = "http://localhost:8000/predict";

    @PostMapping("/api/predict")
    public ResponseEntity<?> predict(@RequestParam("file") MultipartFile file) {
        // 1. 校验文件
        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body("{\"detail\":\"请选择非空的图片文件！\"}");
        }
        // 校验文件类型
        String contentType = file.getContentType();
        if (contentType == null || !contentType.startsWith("image/")) {
            return ResponseEntity.badRequest().body("{\"detail\":\"仅支持上传图片文件（jpg/png/jpeg）！\"}");
        }

        try {
            // 2. 构造请求头（multipart/form-data）
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            // 3. 构造请求体（封装上传文件）
            MultiValueMap<String, Object> requestBody = new LinkedMultiValueMap<>();
            requestBody.add("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename(); // 必须返回文件名，否则FastAPI识别不到
                }
            });

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);

            // 4. 调用FastAPI接口（捕获所有RestTemplate异常）
            ResponseEntity<String> response;
            try {
                response = restTemplate.exchange(
                        MODEL_API_URL,
                        HttpMethod.POST,
                        requestEntity,
                        String.class
                );
            } catch (HttpStatusCodeException e) {
                // 捕获FastAPI返回的500/400等错误（关键！之前没处理）
                String errorBody = e.getResponseBodyAsString();
                return ResponseEntity.status(e.getStatusCode()).body(errorBody);
            } catch (Exception e) {
                // 捕获连接超时/服务未启动等异常
                String errorMsg = "{\"detail\":\"调用Python模型服务失败：" + e.getMessage() + "\"}";
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorMsg);
            }

            // 5. 转发FastAPI的响应给前端（保持原格式）
            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_JSON) // 明确返回JSON格式
                    .body(response.getBody());

        } catch (IOException e) {
            // 文件读取异常
            String errorMsg = "{\"detail\":\"文件读取失败：" + e.getMessage() + "\"}";
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorMsg);
        }
    }
}