#!/bin/bash

git config --global user.name 'Isaac Vander Sluis'
git config --global user.email isaacvandersluis@gmail.com

echo 'Git config:'
git config --list

echo -e '\nCurrent commit:'
git log --oneline -n1