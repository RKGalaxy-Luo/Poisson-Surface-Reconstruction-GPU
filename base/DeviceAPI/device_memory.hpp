/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef DEVICE_MEMORY_HPP_
#define DEVICE_MEMORY_HPP_

#include "kernel_containers.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/** \brief @b DeviceMemory class
  *
  * \note This is a BLOB container class with reference counting for GPU memory.
  *
  * \author Anatoly Baksheev
  */

class DeviceMemory
{
    public:
        /** \brief 空构造函数. */
        DeviceMemory();

        /** \brief 析构函数. */
        ~DeviceMemory();

        /** \brief 在GPU内存中分配内部缓存
          * \param sizeBytes_arg: 要分配的内存量
          * */
        DeviceMemory(size_t sizeBytes_arg);

        /** \brief 使用用户分配的内存进行初始化. 在这种情况下禁用引用计数.
          * \param ptr_arg: 缓存指针
          * \param sizeBytes_arg: 缓存大小
          * */
        DeviceMemory(void *ptr_arg, size_t sizeBytes_arg);

        /** \brief 复制构造函数. 只是增加引用计数器. */
        DeviceMemory(const DeviceMemory& other_arg);

        /** \brief 赋值运算符. 只是增加引用计数器. */
        DeviceMemory& operator=(const DeviceMemory& other_arg);

         /** \brief 在GPU内存中分配内部缓冲区。如果内部缓冲区是在函数用新的大小重新创建之前创建的。如果新旧大小相等，它什么也不做.
           * \param sizeBytes_arg: 缓存大小
           * */
        void create(size_t sizeBytes_arg);

        /** \brief 如果需要，减少引用计数器并释放内部缓冲区. */
        void release();

        /** \brief 执行数据复制。如果目标大小不同，它将被重新分配.
          * \param other_arg: 目标容器
          * */
        void copyTo(DeviceMemory& other) const;

        /** \brief 上传数据到GPU内存的内部缓冲区. 它在内部调用create()函数以确保内部缓冲区大小足够.
          * \param host_ptr_arg: 指向要上传的host缓存的指针
          * \param sizeBytes_arg: 缓存大小
          * */
        void upload(const void *host_ptr_arg, size_t sizeBytes_arg);

        /** \brief 从内部缓冲区下载数据到CPU内存
          * \param host_ptr_arg: 指向要下载的缓冲区的指针
          * */
        void download(void *host_ptr_arg) const;

        /** \brief 执行指向另一个设备内存的数据交换.
          * \param other: 要交换的设备内存
          * */
        void swap(DeviceMemory& other_arg);

        /** \brief 返回GPU内存中内部缓冲区的指针. */
        template<class T> T* ptr();

        /** \brief 返回GPU内存中内部缓冲区的常量指针. */
        template<class T> const T* ptr() const;

        /** \brief 转换为PtrSz以传递给内核函数. */
        template <class U> operator PtrSzPCL<U>() const;

        /** \brief 如果未分配则返回true，否则返回false. */
        bool empty() const;

        /** \brief 返回数据占用GPU缓存的大小. */
        size_t sizeBytes() const;

    private:
        /** \brief 指向GPU中数据的指针. */
        void *data_;

        /** \brief 已分配的内存大小（单位Byte）. */
        size_t sizeBytes_;

        /** \brief 指向CPU内存中引用计数器的指针. */
        int* refcount_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/** \brief @b DeviceMemory2D class
  *
  * \note 这是一个带有引用计数器的将2D的内存数据存入GPU的类.
  *
  * \author Anatoly Baksheev
  */

class DeviceMemory2D
{
    public:
        /** \brief 空的构造函数. */
        DeviceMemory2D();

        /** \brief 析构函数. */
        ~DeviceMemory2D();

        /** \brief 在GPU内存中分配内部缓冲区
          * \param rows_arg: 要分配的行数
          * \param colsBytes_arg: 缓存的宽度，以字节为单位
          * */
        DeviceMemory2D(int rows_arg, int colsBytes_arg);


        /** \brief 使用用户分配的缓存初始化。在这种情况下禁用引用计数.
          * \param rows_arg: 缓存行数
          * \param colsBytes_arg: 缓存的宽度，以字节为单位
          * \param data_arg: 指向缓存的指针
          * \param stepBytes_arg: 在以字节为单位的两个连续行之间步长
          * */
        DeviceMemory2D(int rows_arg, int colsBytes_arg, void *data_arg, size_t step_arg);

        /** \brief 拷贝构造函数. 只是增加引用计数器. */
        DeviceMemory2D(const DeviceMemory2D& other_arg);

        /** \brief 赋值运算符. 只是增加引用计数器. */
        DeviceMemory2D& operator=(const DeviceMemory2D& other_arg);

        /** \brief 在GPU内存中分配内部缓冲区。如果内部缓冲区是在函数用新的大小重新创建之前创建的。如果新旧大小相等，它什么也不做.
           * \param ptr_arg: 要分配的行数
           * \param sizeBytes_arg: 缓存的宽度，以字节为单位
           * */
        void create(int rows_arg, int colsBytes_arg);

        /** \brief 如果需要，减少引用计数器并释放内部缓冲区. */
        void release();

        /** \brief 执行数据复制。如果目标大小不同，它将被重新分配.
          * \param other_arg: 需要拷贝到的容器
          * */
        void copyTo(DeviceMemory2D& other) const;

        /** \brief 上传数据到GPU内存的内部缓冲区。它在内部调用create()函数以确保内部缓冲区大小足够.
          * \param host_ptr_arg: 指向要上传的主机缓冲区的指针
          * \param host_step_arg: 主机缓存中连续两行之间的步长(以字节为单位)
          * \param rows_arg: 要上传的行数
          * \param sizeBytes_arg: 主机缓存的宽度，以字节为单位
          * */
        void upload(const void *host_ptr_arg, size_t host_step_arg, int rows_arg, int colsBytes_arg);

        /** \brief 从内部缓冲区下载数据到CPU内存. 用户负责设置正确的主机缓冲区大小.
          * \param host_ptr_arg: 指向要下载的主机缓冲区的指针
          * \param host_step_arg: 主机缓存中连续两行之间的步长(以字节为单位)
          * */
        void download(void *host_ptr_arg, size_t host_step_arg) const;

        /** \brief 执行指向另一个设备内存的数据交换.
          * \param other: 要交换的设备内存
          * */
        void swap(DeviceMemory2D& other_arg);

        /** \brief 返回指向内部缓冲区中给定行的指针.
          * \param y_arg: 行索引（第几行，默认第0行）
          * */
        template<class T> T* ptr(int y_arg = 0);

        /** \brief 返回指向内部缓存中给定行的常量指针.
          * \param y_arg: 行索引
          * */
        template<class T> const T* ptr(int y_arg = 0) const;

         /** \brief 转换为PtrStep以传递给内核函数. */
        template <class U> operator PtrStepPCL<U>() const;

        /** \brief 转换为PtrStepSzPCL以传递给内核函数. */
        template <class U> operator PtrStepSzPCL<U>() const;

        /** \brief 如果未分配则返回true，否则返回false. */
        bool empty() const;

        /** \brief 返回每行中的字节数. */
        int colsBytes() const;

        /** \brief 返回行数. */
        int rows() const;

        /** \brief 以字节为单位返回两个连续行之间的步幅，用于内部缓存. Step总是以字节的形式存储在任何地方!!! */
        size_t step() const;
    private:
        /** \brief Device pointer. */
        void *data_;

        /** \brief 以字节为单位返回两个连续行之间的步幅，用于内部缓存. Step总是以字节的形式存储在任何地方!!! */
        size_t step_;

        /** \brief 缓存宽度，以字节为单位. */
        int colsBytes_;

        /** \brief 行数. */
        int rows_;

        /** \brief 指向CPU内存中引用计数器的指针. */
        int* refcount_;
};

#include "device_memory_impl.hpp"

#endif /* DEVICE_MEMORY_HPP_ */
