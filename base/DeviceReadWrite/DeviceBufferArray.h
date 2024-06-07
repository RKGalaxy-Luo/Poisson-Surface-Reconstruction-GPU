/*****************************************************************//**
 * \file   DeviceArrayBuffer.h
 * \brief  管理GPU内存，包括只读类型，读写类型
 * 
 * \author LUO
 * \date   January 31st 2024
 *********************************************************************/
#pragma once
#include <base/CommonUtils.h>
#include <base/CommonTypes.h>
#include <base/Logging.h>
#include "DeviceArrayView.h"
#include "DeviceArrayHandle.h"


namespace SparseSurfelFusion{
	template<typename T>
	class DeviceBufferArray {
	private:
		DeviceArray<T> buffer;		// array的首地址，这个相当于是一个缓冲区，而array才是数据区
		DeviceArray<T> array;		// 设备上的数组（根据大小分配内存了）

	public:
		explicit DeviceBufferArray() : buffer(nullptr, 0), array(nullptr, 0) {}
		//
		explicit DeviceBufferArray(size_t capacity) {
			AllocateBuffer(capacity);
			array = DeviceArray<T>(buffer.ptr(), 0);//这里分配了GPU缓存，元素数量为0
		}
		~DeviceBufferArray() = default;

		//没有隐式复制/分配/移动
		NO_COPY_ASSIGN_MOVE(DeviceBufferArray);

		//访问方法
		DeviceArray<T> Array() const { return array; }
		DeviceArrayView<T> ArrayView() const { return DeviceArrayView<T>(array.ptr(), array.size()); }
		DeviceArrayView<T> ArrayReadOnly() const { return DeviceArrayView<T>(array.ptr(), array.size()); }
		DeviceArrayHandle<T> ArrayHandle() { return DeviceArrayHandle<T>(array.ptr(), array.size()); }
		DeviceArray<T> Buffer() const { return buffer; }

		//与other里的数据进行交换
		void swap(DeviceBufferArray<float>& other) {
			buffer.swap(other.buffer);
			array.swap(other.array);
		}


		const T* Ptr() const { return buffer.ptr(); }			//转换为原始指针
		T* Ptr() { return buffer.ptr(); }						//转换为原始指针
		operator T* () { return buffer.ptr(); }					//转换为原始指针
		operator const T* () const { return buffer.ptr(); }		//转换为原始指针

		/**
		 * \brief 查询buffer大小.
		 */
		size_t Capacity() const { return buffer.size(); }
		/**
		 * \brief 查询Buffer大小.
		 */
		size_t BufferSize() const { return buffer.size(); }
		/**
		 * \brief 查询Array大小.
		 */
		size_t ArraySize() const { return array.size(); }

		/**
		 * \brief 分配缓存，如果当下缓存大于所需则返回，如果小于所需则开辟.
		 * 
		 * \param capacity 需要分配缓存的容量
		 */
		void AllocateBuffer(size_t capacity) {
			if (buffer.size() > capacity) return;			// 如果DeviceBufferArray能储存的比capacity大，则直接返回
			buffer.create(capacity);						// 如果GPU内存不足，则开辟内存
			array = DeviceArray<T>(buffer.ptr(), 0);		// 将buffer地址给array，并设置元素数量为0
		}
		/**
		 * \brief 释放Buffer.
		 * 
		 */
		void ReleaseBuffer() {
			if (buffer.size() > 0) buffer.release();
		}

		/**
		 * \brief 调整数组，如果需要调整的大小小于buffer，则将缓冲区的值给array，返回true.
		 *		  需要调整的大小大于buffer，若是allocate是true则分配1.5倍的size大小，并把buffer中的数据赋给array，并返回true.
		 *		  如果需要调整的大小小于buffer，并且allocate是false，则无法分配数据给array，返回false.
		 * 
		 * \param size 需要调整array的大小
		 * \param allocate 如果size大于buffer是否进行重新分配内存(默认不重新分配)
		 * \return 分配Array内存是否成功
		 */
		bool ResizeArray(size_t size, bool allocate = false) {
			if (size <= buffer.size()) {
				array = DeviceArray<T>(buffer.ptr(), size);
				return true;
			}
			else if (allocate) {
				const size_t prev_size = array.size();

				//需要复制旧的元素
				DeviceArray<T> old_buffer = buffer;
				buffer.create(static_cast<size_t>(size * 1.5));
				if (prev_size > 0) {
					CHECKCUDA(cudaMemcpy(buffer.ptr(), old_buffer.ptr(), sizeof(T) * prev_size, cudaMemcpyDeviceToDevice));
					old_buffer.release();
				}

				//分配正确的栈空间
				array = DeviceArray<T>(buffer.ptr(), size);
				return true;
			}
			else {
				return false;
			}
		}


		/**
		 * \brief 调整数组大小：预分配Buffer不够直接报错，预分配Buffer够则将array赋上buffer地址，并且开辟一个size大小的GPU(Array)缓存.
		 * 
		 * \param size 需要调整的大小
		 */
		void ResizeArrayOrException(size_t size) {
			if (size > buffer.size()) {
				printf("Buffer 大小 = %zd\n", buffer.size());
				LOGGING(FATAL) << "预分配的缓冲区不够";
			}
			//如果预分配的缓存足够。则改变数组的大小
			array = DeviceArray<T>(buffer.ptr(), size);
		}

	};
}
