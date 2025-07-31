## file System 4

#### 파일 디렉토리 복사와 이동

```java
Path src = Paths.get(C:\\java\\copy.java);
Path dst = Paths.get(C:\\java\\copy2.java);


Files.copy(src, dst, StandardCopyOption.REPLACE_EXISTING);
```