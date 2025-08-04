## wsl(Windows Subsystem for Linux)

#### 윈도우 안에서 리눅스를 가상머신처럼 무겁지 않게 돌릴 수 있게 해주는 기능

- 설치
- 제어판 > 프로그램 > 프로그램 및 기능 > Windows 기능 켜기 또는 끄기
- Linux용 윈도우 하위시스템 켜기, 가상 머신 플랫폼 켜기
- 재부팅 후 Microsoft Store에서 Ubuntu를 검색하고 22.04를 선택하여 설치
- cmd.exe 실행
- wsl --install Ubuntu-22.04로 설치, ubuntu 계정 생성하고 암호 설정
- wsl -l -v로 현재 설치된 배포본 확인
- wsl –d Ubuntu-22.04로 wsl 콘솔 진입
- 종료는 wsl --terminate Ubuntu-22.04
- 배포본 삭제는 wsl --unregister Ubuntu-22.04
  
#### 이후 git 계정 생성하고 wsl 로그인 및 작업 디렉토리 생성

- git 계정 준비
- mkdir ~/work
- cd ~/work
- git config 설정을 해준다.(username, email)
- ssh 키 설정 및 git clone
- ssh-keygen을 통해 rsa, rsa.pub 키 발급 : cat ~/.ssh/id_rsa.pub
- rsa.pub 키를 git settings에 토큰 등록하듯이 등록
- git clone "원하는 깃허브 주소"