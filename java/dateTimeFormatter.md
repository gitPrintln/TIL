## DateTimeFormatter

#### 날짜와 시각 정보의 출력 포맷 지정

```java
DateTimeFormatter fm1 = DateTimeFormatter.ofPattern("yy-M-d"); // 19-4-5
DateTimeFormatter fm1 = DateTimeFormatter.ofPattern("yyyy-MM-d, H:m:s"); // 2025-07-21, 10:10:0
DateTimeFormatter fm1 = DateTimeFormatter.ofPattern("yyyy-MM-d, HH:mm:ss VV"); // 2025-07-21, 10:10:00 Asia/Seoul
```