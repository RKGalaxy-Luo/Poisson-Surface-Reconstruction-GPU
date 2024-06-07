/*****************************************************************//**
 * \file   DeviceArrayBuffer.h
 * \brief  ����GPU�ڴ棬����ֻ�����ͣ���д����
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
		DeviceArray<T> buffer;		// array���׵�ַ������൱����һ������������array����������
		DeviceArray<T> array;		// �豸�ϵ����飨���ݴ�С�����ڴ��ˣ�

	public:
		explicit DeviceBufferArray() : buffer(nullptr, 0), array(nullptr, 0) {}
		//
		explicit DeviceBufferArray(size_t capacity) {
			AllocateBuffer(capacity);
			array = DeviceArray<T>(buffer.ptr(), 0);//���������GPU���棬Ԫ������Ϊ0
		}
		~DeviceBufferArray() = default;

		//û����ʽ����/����/�ƶ�
		NO_COPY_ASSIGN_MOVE(DeviceBufferArray);

		//���ʷ���
		DeviceArray<T> Array() const { return array; }
		DeviceArrayView<T> ArrayView() const { return DeviceArrayView<T>(array.ptr(), array.size()); }
		DeviceArrayView<T> ArrayReadOnly() const { return DeviceArrayView<T>(array.ptr(), array.size()); }
		DeviceArrayHandle<T> ArrayHandle() { return DeviceArrayHandle<T>(array.ptr(), array.size()); }
		DeviceArray<T> Buffer() const { return buffer; }

		//��other������ݽ��н���
		void swap(DeviceBufferArray<float>& other) {
			buffer.swap(other.buffer);
			array.swap(other.array);
		}


		const T* Ptr() const { return buffer.ptr(); }			//ת��Ϊԭʼָ��
		T* Ptr() { return buffer.ptr(); }						//ת��Ϊԭʼָ��
		operator T* () { return buffer.ptr(); }					//ת��Ϊԭʼָ��
		operator const T* () const { return buffer.ptr(); }		//ת��Ϊԭʼָ��

		/**
		 * \brief ��ѯbuffer��С.
		 */
		size_t Capacity() const { return buffer.size(); }
		/**
		 * \brief ��ѯBuffer��С.
		 */
		size_t BufferSize() const { return buffer.size(); }
		/**
		 * \brief ��ѯArray��С.
		 */
		size_t ArraySize() const { return array.size(); }

		/**
		 * \brief ���仺�棬������»�����������򷵻أ����С�������򿪱�.
		 * 
		 * \param capacity ��Ҫ���仺�������
		 */
		void AllocateBuffer(size_t capacity) {
			if (buffer.size() > capacity) return;			// ���DeviceBufferArray�ܴ���ı�capacity����ֱ�ӷ���
			buffer.create(capacity);						// ���GPU�ڴ治�㣬�򿪱��ڴ�
			array = DeviceArray<T>(buffer.ptr(), 0);		// ��buffer��ַ��array��������Ԫ������Ϊ0
		}
		/**
		 * \brief �ͷ�Buffer.
		 * 
		 */
		void ReleaseBuffer() {
			if (buffer.size() > 0) buffer.release();
		}

		/**
		 * \brief �������飬�����Ҫ�����Ĵ�СС��buffer���򽫻�������ֵ��array������true.
		 *		  ��Ҫ�����Ĵ�С����buffer������allocate��true�����1.5����size��С������buffer�е����ݸ���array��������true.
		 *		  �����Ҫ�����Ĵ�СС��buffer������allocate��false�����޷��������ݸ�array������false.
		 * 
		 * \param size ��Ҫ����array�Ĵ�С
		 * \param allocate ���size����buffer�Ƿ�������·����ڴ�(Ĭ�ϲ����·���)
		 * \return ����Array�ڴ��Ƿ�ɹ�
		 */
		bool ResizeArray(size_t size, bool allocate = false) {
			if (size <= buffer.size()) {
				array = DeviceArray<T>(buffer.ptr(), size);
				return true;
			}
			else if (allocate) {
				const size_t prev_size = array.size();

				//��Ҫ���ƾɵ�Ԫ��
				DeviceArray<T> old_buffer = buffer;
				buffer.create(static_cast<size_t>(size * 1.5));
				if (prev_size > 0) {
					CHECKCUDA(cudaMemcpy(buffer.ptr(), old_buffer.ptr(), sizeof(T) * prev_size, cudaMemcpyDeviceToDevice));
					old_buffer.release();
				}

				//������ȷ��ջ�ռ�
				array = DeviceArray<T>(buffer.ptr(), size);
				return true;
			}
			else {
				return false;
			}
		}


		/**
		 * \brief ���������С��Ԥ����Buffer����ֱ�ӱ���Ԥ����Buffer����array����buffer��ַ�����ҿ���һ��size��С��GPU(Array)����.
		 * 
		 * \param size ��Ҫ�����Ĵ�С
		 */
		void ResizeArrayOrException(size_t size) {
			if (size > buffer.size()) {
				printf("Buffer ��С = %zd\n", buffer.size());
				LOGGING(FATAL) << "Ԥ����Ļ���������";
			}
			//���Ԥ����Ļ����㹻����ı�����Ĵ�С
			array = DeviceArray<T>(buffer.ptr(), size);
		}

	};
}
