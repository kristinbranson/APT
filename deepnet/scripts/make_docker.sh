docker build -t bransonlabapt/apt_docker:tf1.15_py3 -f dockerfile_tf115_py3 .
docker tag bransonlabapt/apt_docker:tf1.15_py3 bransonlabapt/apt_docker:latest
docker build -t bransonlabapt/apt_docker:tf1.15_py3_cpu -f dockerfile_tf115_py3_cpu .
docker tag bransonlabapt/apt_docker:tf1.15_py3_cpu bransonlabapt/apt_docker:latest_cpu
docker push  bransonlabapt/apt_docker:tf1.15_py3_cpu
docker push  bransonlabapt/apt_docker:tf1.15_py3
# Tf 113 and py2
#docker build -t bransonlabapt/apt_docker:tf1.13 -f dockerfile_tf113 .
#docker tag bransonlabapt/apt_docker:tf1.13 bransonlabapt/apt_docker:latest
#docker build -t bransonlabapt/apt_docker:tf1.13_cpu -f dockerfile_tf113_cpu .
#docker tag bransonlabapt/apt_docker:tf1.13_cpu bransonlabapt/apt_docker:latest_cpu
