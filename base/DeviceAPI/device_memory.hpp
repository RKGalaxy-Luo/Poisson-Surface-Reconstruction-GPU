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
        /** \brief �չ��캯��. */
        DeviceMemory();

        /** \brief ��������. */
        ~DeviceMemory();

        /** \brief ��GPU�ڴ��з����ڲ�����
          * \param sizeBytes_arg: Ҫ������ڴ���
          * */
        DeviceMemory(size_t sizeBytes_arg);

        /** \brief ʹ���û�������ڴ���г�ʼ��. ����������½������ü���.
          * \param ptr_arg: ����ָ��
          * \param sizeBytes_arg: �����С
          * */
        DeviceMemory(void *ptr_arg, size_t sizeBytes_arg);

        /** \brief ���ƹ��캯��. ֻ���������ü�����. */
        DeviceMemory(const DeviceMemory& other_arg);

        /** \brief ��ֵ�����. ֻ���������ü�����. */
        DeviceMemory& operator=(const DeviceMemory& other_arg);

         /** \brief ��GPU�ڴ��з����ڲ�������������ڲ����������ں������µĴ�С���´���֮ǰ�����ġ�����¾ɴ�С��ȣ���ʲôҲ����.
           * \param sizeBytes_arg: �����С
           * */
        void create(size_t sizeBytes_arg);

        /** \brief �����Ҫ���������ü��������ͷ��ڲ�������. */
        void release();

        /** \brief ִ�����ݸ��ơ����Ŀ���С��ͬ�����������·���.
          * \param other_arg: Ŀ������
          * */
        void copyTo(DeviceMemory& other) const;

        /** \brief �ϴ����ݵ�GPU�ڴ���ڲ�������. �����ڲ�����create()������ȷ���ڲ���������С�㹻.
          * \param host_ptr_arg: ָ��Ҫ�ϴ���host�����ָ��
          * \param sizeBytes_arg: �����С
          * */
        void upload(const void *host_ptr_arg, size_t sizeBytes_arg);

        /** \brief ���ڲ��������������ݵ�CPU�ڴ�
          * \param host_ptr_arg: ָ��Ҫ���صĻ�������ָ��
          * */
        void download(void *host_ptr_arg) const;

        /** \brief ִ��ָ����һ���豸�ڴ�����ݽ���.
          * \param other: Ҫ�������豸�ڴ�
          * */
        void swap(DeviceMemory& other_arg);

        /** \brief ����GPU�ڴ����ڲ���������ָ��. */
        template<class T> T* ptr();

        /** \brief ����GPU�ڴ����ڲ��������ĳ���ָ��. */
        template<class T> const T* ptr() const;

        /** \brief ת��ΪPtrSz�Դ��ݸ��ں˺���. */
        template <class U> operator PtrSzPCL<U>() const;

        /** \brief ���δ�����򷵻�true�����򷵻�false. */
        bool empty() const;

        /** \brief ��������ռ��GPU����Ĵ�С. */
        size_t sizeBytes() const;

    private:
        /** \brief ָ��GPU�����ݵ�ָ��. */
        void *data_;

        /** \brief �ѷ�����ڴ��С����λByte��. */
        size_t sizeBytes_;

        /** \brief ָ��CPU�ڴ������ü�������ָ��. */
        int* refcount_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/** \brief @b DeviceMemory2D class
  *
  * \note ����һ���������ü������Ľ�2D���ڴ����ݴ���GPU����.
  *
  * \author Anatoly Baksheev
  */

class DeviceMemory2D
{
    public:
        /** \brief �յĹ��캯��. */
        DeviceMemory2D();

        /** \brief ��������. */
        ~DeviceMemory2D();

        /** \brief ��GPU�ڴ��з����ڲ�������
          * \param rows_arg: Ҫ���������
          * \param colsBytes_arg: ����Ŀ�ȣ����ֽ�Ϊ��λ
          * */
        DeviceMemory2D(int rows_arg, int colsBytes_arg);


        /** \brief ʹ���û�����Ļ����ʼ��������������½������ü���.
          * \param rows_arg: ��������
          * \param colsBytes_arg: ����Ŀ�ȣ����ֽ�Ϊ��λ
          * \param data_arg: ָ�򻺴��ָ��
          * \param stepBytes_arg: �����ֽ�Ϊ��λ������������֮�䲽��
          * */
        DeviceMemory2D(int rows_arg, int colsBytes_arg, void *data_arg, size_t step_arg);

        /** \brief �������캯��. ֻ���������ü�����. */
        DeviceMemory2D(const DeviceMemory2D& other_arg);

        /** \brief ��ֵ�����. ֻ���������ü�����. */
        DeviceMemory2D& operator=(const DeviceMemory2D& other_arg);

        /** \brief ��GPU�ڴ��з����ڲ�������������ڲ����������ں������µĴ�С���´���֮ǰ�����ġ�����¾ɴ�С��ȣ���ʲôҲ����.
           * \param ptr_arg: Ҫ���������
           * \param sizeBytes_arg: ����Ŀ�ȣ����ֽ�Ϊ��λ
           * */
        void create(int rows_arg, int colsBytes_arg);

        /** \brief �����Ҫ���������ü��������ͷ��ڲ�������. */
        void release();

        /** \brief ִ�����ݸ��ơ����Ŀ���С��ͬ�����������·���.
          * \param other_arg: ��Ҫ������������
          * */
        void copyTo(DeviceMemory2D& other) const;

        /** \brief �ϴ����ݵ�GPU�ڴ���ڲ��������������ڲ�����create()������ȷ���ڲ���������С�㹻.
          * \param host_ptr_arg: ָ��Ҫ�ϴ���������������ָ��
          * \param host_step_arg: ������������������֮��Ĳ���(���ֽ�Ϊ��λ)
          * \param rows_arg: Ҫ�ϴ�������
          * \param sizeBytes_arg: ��������Ŀ�ȣ����ֽ�Ϊ��λ
          * */
        void upload(const void *host_ptr_arg, size_t host_step_arg, int rows_arg, int colsBytes_arg);

        /** \brief ���ڲ��������������ݵ�CPU�ڴ�. �û�����������ȷ��������������С.
          * \param host_ptr_arg: ָ��Ҫ���ص�������������ָ��
          * \param host_step_arg: ������������������֮��Ĳ���(���ֽ�Ϊ��λ)
          * */
        void download(void *host_ptr_arg, size_t host_step_arg) const;

        /** \brief ִ��ָ����һ���豸�ڴ�����ݽ���.
          * \param other: Ҫ�������豸�ڴ�
          * */
        void swap(DeviceMemory2D& other_arg);

        /** \brief ����ָ���ڲ��������и����е�ָ��.
          * \param y_arg: ���������ڼ��У�Ĭ�ϵ�0�У�
          * */
        template<class T> T* ptr(int y_arg = 0);

        /** \brief ����ָ���ڲ������и����еĳ���ָ��.
          * \param y_arg: ������
          * */
        template<class T> const T* ptr(int y_arg = 0) const;

         /** \brief ת��ΪPtrStep�Դ��ݸ��ں˺���. */
        template <class U> operator PtrStepPCL<U>() const;

        /** \brief ת��ΪPtrStepSzPCL�Դ��ݸ��ں˺���. */
        template <class U> operator PtrStepSzPCL<U>() const;

        /** \brief ���δ�����򷵻�true�����򷵻�false. */
        bool empty() const;

        /** \brief ����ÿ���е��ֽ���. */
        int colsBytes() const;

        /** \brief ��������. */
        int rows() const;

        /** \brief ���ֽ�Ϊ��λ��������������֮��Ĳ����������ڲ�����. Step�������ֽڵ���ʽ�洢���κεط�!!! */
        size_t step() const;
    private:
        /** \brief Device pointer. */
        void *data_;

        /** \brief ���ֽ�Ϊ��λ��������������֮��Ĳ����������ڲ�����. Step�������ֽڵ���ʽ�洢���κεط�!!! */
        size_t step_;

        /** \brief �����ȣ����ֽ�Ϊ��λ. */
        int colsBytes_;

        /** \brief ����. */
        int rows_;

        /** \brief ָ��CPU�ڴ������ü�������ָ��. */
        int* refcount_;
};

#include "device_memory_impl.hpp"

#endif /* DEVICE_MEMORY_HPP_ */
