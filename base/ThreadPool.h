/*****************************************************************//**
 * \file   ThreadPool.h
 * \brief  构建任务线程池
 * 
 * \author LUO
 * \date   January 18th 2024
 *********************************************************************/
#pragma once
#include <iostream>
#include <functional>
#include <mutex>
#include <vector>
#include <queue>
#include <thread>
#include <condition_variable>
#include <future>
#include "Logging.h"

/**
 * \brief 为了防止多个线程同时访问队列，因此封装的安全队列对象.
 */
template <typename T>
class SafeQueue
{
private:
	std::queue<T> safeQueue; // 利用模板函数构造队列

	std::mutex mutexlock; // 访问互斥信号量	

public:
	/**
	 * \brief 构造安全队列对象，这样的对象会避免多个线程访问同一个队列.
	 * 
	 */
	SafeQueue() {}
	/**
	 * \brief 构造安全队列.
	 * 
	 * \param other 一个安全队列
	 */
	SafeQueue(SafeQueue&& other) {}
	/**
	 * \brief 析构安全队列.
	 * 
	 */
	~SafeQueue() {}

	/**
	 * \brief 获取当前队列是否为空.
	 * 
	 * \return 为空返回true，不为空返回false
	 */
	bool Empty() {
		std::unique_lock<std::mutex> lock(mutexlock);// 互斥信号变量加锁，防止safeQueue被改变
		return safeQueue.empty();
	}

	/**
	 * \brief 返回队列中元素个数.
	 *
	 * \return 元素个数
	 */
	int Size() {
		std::unique_lock<std::mutex> lock(mutexlock); // 互斥信号变量加锁，防止safeQueue被改变
		return safeQueue.size();
	}

	/**
	 * \brief 队列添加元素.
	 *
	 * \param t 添加的元素
	 */
	void Enqueue(T& t) {
		std::unique_lock<std::mutex> lock(mutexlock);
		safeQueue.emplace(t); // emplace既能接收左值，也能接收右值
	}

	/**
	 * \brief 队列取出元素.
	 *
	 * \param t 元素出队
	 * \return
	 */
	bool Dequeue(T& t) {
		std::unique_lock<std::mutex> lock(mutexlock); // 队列加锁
		if (safeQueue.empty())	return false;
		t = std::move(safeQueue.front()); // 取出队首元素，返回队首元素值，并进行右值引用
		safeQueue.pop(); // 弹出入队的第一个元素
		return true;
	}
};

/**
 * \brief 线程池对象.
 */
class ThreadPool
{
private:
	bool stop;										// 是否停止
	SafeQueue<std::function<void()>> tasks;			// 线程池的任务队列，传入线程中执行的函数
	std::vector<std::thread> workers;				// 用来工作的线程，一共多少个线程
	std::condition_variable cond_;					// 控制某线程是否阻塞
	std::mutex mutex_;								// 线程休眠锁互斥变量

private:
	/**
	 * \brief 内置的工作线程类.
	 */
	class ThreadWorker
	{
	private:
		int workerID;		//线程工作的ID
		ThreadPool* threadPool;	//所属线程池
	public:
		ThreadWorker(const int id, ThreadPool* pool) : workerID(id), threadPool(pool) {}
		// 重载()操作
		void operator ()()
		{
			std::function<void()> task; // 定义基础函数类，创建任务task，即需要执行的函数
			bool isDequeued;			// 是否正在取出队列中元素
			while (!threadPool->stop) {	// 线程池没有关闭，死循环等待，下面是判断是否继续循环执行任务的临界区域
				{	// {}：标定lock的作用域
					// 为线程环境加锁，互访问工作线程的休眠和唤醒
					std::unique_lock<std::mutex> lock(threadPool->mutex_);
					if (threadPool->tasks.Empty())		// 如果任务队列为空，阻塞当前线程
					{
						threadPool->cond_.wait(lock);	// 等待条件变量通知，开启线程
					}
					// 取出任务队列中的元素
					isDequeued = threadPool->tasks.Dequeue(task);	//将队列的头部移动给task
				}	// 临界区结束，自动释放锁 mutex_
				if (isDequeued) task();		//出队成功执行task任务
			}
		}
	};


public:

	/**
	 * \brief 构造线程池初始化并分配线程.
	 * 
	 * \param size 构造的线程池有多少个线程
	 */
	ThreadPool(const int threadNum) : workers(std::vector<std::thread>(threadNum)), stop(false) {
		for (int i = 0; i < threadNum; i++) { // 分配线程
			workers.at(i) = std::thread(ThreadWorker(i, this));
		}
	}

	ThreadPool(const ThreadPool &)=delete;

	ThreadPool(ThreadPool &&)=delete;

	ThreadPool &operator=(const ThreadPool &)=delete;

	ThreadPool &&operator=(ThreadPool &&)=delete;

	/**
	 * \brief 通知线程池里所有的线程准备析构，停止任务进队列，阻塞到所有的任务完成.
	 * 
	 */
	inline ~ThreadPool() {
		
		std::unique_lock<std::mutex> lock(mutex_);		// 临界区开始  锁住stop 将其赋值为true 通知线程退出循环
		stop = true;									// 临界区结束 自动释放锁 mutex_
		cond_.notify_all();								// 通知所有的线程退出循环
		for (int i = 0; i < workers.size(); i++) {		// 等待直到所有线程结束
			if (workers.at(i).joinable()) {
				workers.at(i).join();
			}
		}
	}

	///**
	// * \brief 通过完美转发进行任务入队列. C++17
	// * 
	// * \param 传入需要加入线程池的函数
	// * \param ...args 
	// * \return 
	// */
	//template<class F, class ... Args>			// 类型可推导，通用模板
	//auto AddTask(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {	// 用auto将返回值后置,用future在未来获得函数类型
	//	using returnType = std::future<typename std::invoke_result<F, Args...>::type>;// 用推导得到返回值类型

	//	std::function<typename std::invoke_result<F, Args...>::type()> func =
	//		std::bind(std::forward<F>(f), std::forward<Args>(args)...);// 连接函数和参数定义，特殊函数类型，避免左右值错误
	//	// 用packageed_task包装一个函数对象，可以异步调用这个函数对象，就是吧一个普通函数对象转成异步执行的任务
	//	auto task = std::make_shared< std::packaged_task<typename std::invoke_result<F, Args...>::type()> >( func );

	//	std::function<void()> warpperFunction = [task]() {(*task)(); };
	//	returnType res = task->get_future();
	//	{
	//		//临界区开始 锁任务队列
	//		std::unique_lock<std::mutex> lock(mutex_);
	//		if (stop) LOGGING(FATAL) << "线程已经停止，不允许入队";
	//		//如果已经将线程池析构,就不允许再入队列
	//		tasks.Enqueue(warpperFunction);
	//	}
	//	cond_.notify_one(); //通知任意一个线程接收任务
	//	return res;
	//}

	/**
	 * \brief 通过完美转发进行任务入队列. C++14
	 *
	 * \param 传入需要加入线程池的函数
	 * \param ...args
	 * \return
	 */
	template<class F, class ... Args>			// 类型可推导，通用模板
	auto AddTask(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {	// 用auto将返回值后置,用future在未来获得函数类型
		using returnType = std::future<typename std::result_of<F(Args...)>::type>;// 用推导得到返回值类型

		std::function<typename std::result_of<F(Args...)>::type()> func =
			std::bind(std::forward<F>(f), std::forward<Args>(args)...);// 连接函数和参数定义，特殊函数类型，避免左右值错误
		// 用packageed_task包装一个函数对象，可以异步调用这个函数对象，就是吧一个普通函数对象转成异步执行的任务
		auto task = std::make_shared< std::packaged_task<typename std::result_of<F(Args...)>::type()> >(func);

		std::function<void()> warpperFunction = [task]() {(*task)(); };
		returnType res = task->get_future();
		{
			//临界区开始 锁任务队列
			std::unique_lock<std::mutex> lock(mutex_);
			if (stop) LOGGING(FATAL) << "线程已经停止，不允许入队";
			//如果已经将线程池析构,就不允许再入队列
			tasks.Enqueue(warpperFunction);
		}
		cond_.notify_one(); //通知任意一个线程接收任务
		return res;
	}
};

