/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <vector>

template <class T>
class CTensor
{
  public:
    T *data;
    std::vector<std::size_t> shape;
    std::size_t data_size;

    CTensor(T * _data, std::vector<std::size_t> _shape)
    {
        this->shape = _shape;
        std::size_t _data_size = 1;
        for (int data_i = 0; data_i < shape.size(); data_i++)
        {
            _data_size *= shape[data_i];
        }
        this->data_size = _data_size;
        this->data = _data;
    }

    CTensor( std::vector<std::size_t> _shape)
    {
        this->shape = _shape;
        std::size_t _data_size = 1;
        for (int data_i = 0; data_i < shape.size(); data_i++)
        {
            _data_size *= shape[data_i];
        }
        this->data_size = _data_size;
        this->data = new T[_data_size];
    }

    void printCTensor(){
        std::cout<<"Data Shape:"<<std::endl;
        for(int i=0;i<this->shape.size();i++){
            std::cout<<this->shape[i]<<",";
        }
        std::cout<<std::endl<<"Data Size:"<<this->data_size<<std::endl;
    }
    ~CTensor(){ }
  private:
};