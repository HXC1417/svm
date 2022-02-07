# project name #定义项目名称
PROJECT(photo) #找到这个项目名称

cmake_minimum_required(VERSION 3.5) #连接高版本和低版本的工具

# gjhhhj
find_package(OpenCV REQUIRED) #检查引入库的路径


include_directories(${OpenCV_INCLUDE_DIRS}) #设置链接库搜索目录库

add_executable(photo main.cpp) #编译可执行程序
# .so

link_directories(${OpenCV_LIBRARY_DIRS}) #链接这个目录
target_link_libraries(photo ${OpenCV_LIBRARIES}) #最终输出可执行文件的命名
