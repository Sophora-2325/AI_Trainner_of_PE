# Webots Makefile for Walkforward controller using ROBOTIS OP2 libraries

null :=
space := $(null) $(null)
WEBOTS_HOME_PATH?=$(subst $(space),\ ,$(strip $(subst \,/,$(WEBOTS_HOME))))
RESOURCES_PATH = $(WEBOTS_HOME)/projects/robots/robotis/darwin-op
INCLUDE = -I"$(RESOURCES_PATH)/libraries/managers/include" -I"$(RESOURCES_PATH)/libraries/robotis-op2/robotis/Framework/include"
LIBRARIES = -L"$(RESOURCES_PATH)/libraries/robotis-op2" -lrobotis-op2 -L"$(RESOURCES_PATH)/libraries/managers" -lmanagers
CXX_SOURCES = $(wildcard *.cpp)

include $(WEBOTS_HOME_PATH)/resources/Makefile.include
