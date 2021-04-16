#!/bin/bash

ALPHABET=( {a..z} )

for ((i=0; i<26; i++))
do
  echo "========================= alphabet ${ALPHABET[i]} ========================="
  python -m torch2trt.tests.test --tolerance 0.0001 --alphabet ${ALPHABET[i]}
  echo 
done