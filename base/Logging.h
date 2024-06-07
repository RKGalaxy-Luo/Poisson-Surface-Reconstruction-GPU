#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <ctime>
#include <cassert>
#include <fstream>
#include <sstream>
#include <chrono>


namespace SparseSurfelFusion {
	struct LogCheckError {
		LogCheckError() : str(nullptr) {}
		explicit LogCheckError(const std::string& str_p) : str(new std::string(str_p)) {}
		~LogCheckError() {
			if (str != nullptr) delete str;
		}
		//Type conversion
		operator bool() { return str != nullptr; }
		//The error string
		std::string* str;
	};

#define DEFINE_CHECK_FUNCTION(name, op)								\
	template <typename X, typename Y>                               \
	inline LogCheckError LogCheck##name(const X& x, const Y& y) {   \
		if (x op y) return LogCheckError();                         \
        std::ostringstream os;                                      \
        os << " (" << x << " vs. " << y << ") ";                    \
        return LogCheckError(os.str());                             \
	}                                                               \
    inline LogCheckError LogCheck##name(int x, int y) {             \
        return LogCheck##name<int, int>(x, y);                      \
    }
		DEFINE_CHECK_FUNCTION(_LT, < )
		DEFINE_CHECK_FUNCTION(_GT, > )
		DEFINE_CHECK_FUNCTION(_LE, <= )
		DEFINE_CHECK_FUNCTION(_GE, >= )
		DEFINE_CHECK_FUNCTION(_EQ, == )
		DEFINE_CHECK_FUNCTION(_NE, != )



		//���Ǽ��
#define FUNCTION_CHECK(x)																	\
	if(!(x))																				\
        SparseSurfelFusion::LogMessageFatal(__FILE__, __LINE__).stream()					\
        << "���ʧ��: " #x << " "

#define FUNCTION_CHECK_BINARY_OP(name, op, x, y)											\
	if(SparseSurfelFusion::LogCheckError err = SparseSurfelFusion::LogCheck##name(x, y))	\
        SparseSurfelFusion::LogMessageFatal(__FILE__, __LINE__).stream()					\
	    << "���ʧ��: " << #x " " #op " " #y << *(err.str)

#define	FUNCTION_CHECK_LT(x, y) FUNCTION_CHECK_BINARY_OP(_LT, < , x, y) //���С��
#define	FUNCTION_CHECK_GT(x, y) FUNCTION_CHECK_BINARY_OP(_GT, > , x, y)	//������
#define	FUNCTION_CHECK_LE(x, y) FUNCTION_CHECK_BINARY_OP(_LE, <=, x, y)	//���С�ڵ���
#define	FUNCTION_CHECK_GE(x, y) FUNCTION_CHECK_BINARY_OP(_GE, >=, x, y)	//�����ڵ���
#define	FUNCTION_CHECK_EQ(x, y) FUNCTION_CHECK_BINARY_OP(_EQ, ==, x, y)	//������
#define	FUNCTION_CHECK_NE(x, y) FUNCTION_CHECK_BINARY_OP(_NE, !=, x, y)	//��鲻����

//���Ժ�ʹ�õ���־����
#define LOGGING_INFO SparseSurfelFusion::LogMessage(__FILE__, __LINE__)			//��ʾ��Ϣ
#define LOGGING_ERROR LOG_INFO		//��ʾ��Ϣ
#define LOGGING_WARNING LOG_INFO	//��ʾ��Ϣ
#define LOGGING_FATAL SparseSurfelFusion::LogMessageFatal(__FILE__, __LINE__)	//��������
#define LOGGING_BEFORE_THROW SparseSurfelFusion::LogMessage().stream()			//�׳��쳣�������

//���ڲ�ͬ�����س̶�
#define LOGGING(severity) LOGGING_##severity.stream()

	// ��־����
	class LogMessage {
	public:
		//���캯��
		LogMessage() : log_stream_(std::cout) {}
		LogMessage(const char* file, int line) : log_stream_(std::cout) {
			log_stream_ << file << ":" << line << ": ";
		}
		LogMessage(const LogMessage&) = delete;
		LogMessage& operator=(const LogMessage&) = delete;

		//����
		~LogMessage() { log_stream_ << "\n"; }

		std::ostream& stream() { return log_stream_; }
	protected:
		std::ostream& log_stream_;
	};

	//����������־����
	class LogMessageFatal {
	public:
		LogMessageFatal(const char* file, int line) {
			log_stream_ << file << ":" << line << ": ";

		}

		//û�и���/����
		LogMessageFatal(const LogMessageFatal&) = delete;
		LogMessageFatal& operator=(LogMessageFatal&) = delete;

		//ʹ����ϵͳʧЧ
		~LogMessageFatal() {
			//�������������������׳��쳣
			throwErrorInfo();
		}
		//����ַ�����
		std::ostringstream& stream() { return log_stream_; }
	protected:
		std::ostringstream log_stream_;

	private:
		/**
		 * �׳������쳣����.
		 * 
		 */
		void throwErrorInfo() {
			LOGGING_BEFORE_THROW << log_stream_.str();
			throw new std::runtime_error(log_stream_.str());
		}
	};

}