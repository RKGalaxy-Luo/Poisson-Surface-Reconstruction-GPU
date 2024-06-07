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



		//总是检查
#define FUNCTION_CHECK(x)																	\
	if(!(x))																				\
        SparseSurfelFusion::LogMessageFatal(__FILE__, __LINE__).stream()					\
        << "检查失败: " #x << " "

#define FUNCTION_CHECK_BINARY_OP(name, op, x, y)											\
	if(SparseSurfelFusion::LogCheckError err = SparseSurfelFusion::LogCheck##name(x, y))	\
        SparseSurfelFusion::LogMessageFatal(__FILE__, __LINE__).stream()					\
	    << "检查失败: " << #x " " #op " " #y << *(err.str)

#define	FUNCTION_CHECK_LT(x, y) FUNCTION_CHECK_BINARY_OP(_LT, < , x, y) //检查小于
#define	FUNCTION_CHECK_GT(x, y) FUNCTION_CHECK_BINARY_OP(_GT, > , x, y)	//检查大于
#define	FUNCTION_CHECK_LE(x, y) FUNCTION_CHECK_BINARY_OP(_LE, <=, x, y)	//检查小于等于
#define	FUNCTION_CHECK_GE(x, y) FUNCTION_CHECK_BINARY_OP(_GE, >=, x, y)	//检查大于等于
#define	FUNCTION_CHECK_EQ(x, y) FUNCTION_CHECK_BINARY_OP(_EQ, ==, x, y)	//检查等于
#define	FUNCTION_CHECK_NE(x, y) FUNCTION_CHECK_BINARY_OP(_NE, !=, x, y)	//检查不等于

//供以后使用的日志类型
#define LOGGING_INFO SparseSurfelFusion::LogMessage(__FILE__, __LINE__)			//提示信息
#define LOGGING_ERROR LOG_INFO		//提示信息
#define LOGGING_WARNING LOG_INFO	//提示信息
#define LOGGING_FATAL SparseSurfelFusion::LogMessageFatal(__FILE__, __LINE__)	//致命错误
#define LOGGING_BEFORE_THROW SparseSurfelFusion::LogMessage().stream()			//抛出异常后的内容

//对于不同的严重程度
#define LOGGING(severity) LOGGING_##severity.stream()

	// 日志内容
	class LogMessage {
	public:
		//构造函数
		LogMessage() : log_stream_(std::cout) {}
		LogMessage(const char* file, int line) : log_stream_(std::cout) {
			log_stream_ << file << ":" << line << ": ";
		}
		LogMessage(const LogMessage&) = delete;
		LogMessage& operator=(const LogMessage&) = delete;

		//换行
		~LogMessage() { log_stream_ << "\n"; }

		std::ostream& stream() { return log_stream_; }
	protected:
		std::ostream& log_stream_;
	};

	//致命错误日志内容
	class LogMessageFatal {
	public:
		LogMessageFatal(const char* file, int line) {
			log_stream_ << file << ":" << line << ": ";

		}

		//没有复制/分配
		LogMessageFatal(const LogMessageFatal&) = delete;
		LogMessageFatal& operator=(LogMessageFatal&) = delete;

		//使整个系统失效
		~LogMessageFatal() {
			//尽量不在析构函数中抛出异常
			throwErrorInfo();
		}
		//输出字符串流
		std::ostringstream& stream() { return log_stream_; }
	protected:
		std::ostringstream log_stream_;

	private:
		/**
		 * 抛出致命异常函数.
		 * 
		 */
		void throwErrorInfo() {
			LOGGING_BEFORE_THROW << log_stream_.str();
			throw new std::runtime_error(log_stream_.str());
		}
	};

}