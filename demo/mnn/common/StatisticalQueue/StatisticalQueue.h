#pragma once
#define SQ_VERSION		(V1.0.13_20221115)

//////////////////////////////////////////////////////////////////////////////////////////
//							class StatisticalQueue 使用说明								//
//      该类用于统计固定时间周期内单帧状态的时间加权累积值或次数累积值。				//
//	    int push(T data); 输入状态或权重，返回累积状态。								//
//	    T getCumulants(); 返回累积值。													//
//	    统计的时间周期采取就近原则：假设cycle=100ms，前后两帧的时长分别是96ms和102ms，	//
//	那么统计周期选择102ms。																//
//	    validPeriod是单帧时间限制参数，在我们的统计规则下，后一帧的状态代表从前后两帧	//
//	之间的状态，由于一些不可预料的原因导致前后帧的时间间隔变大会引起时间的错误统计，	//
//	导致误报。如果前后两帧时间间隔大于validPeriod，则使用validPeriod替代时间间隔。默	//
//	认值0表示不限制。当然该参数只对SQ_FORM_TIME有效，次数统计跟单帧时间无关。			//
//																						//
//  使用案例：																			//
//  1. 急加速判定。																		//
//	   条件：3秒内车辆制动加速度大于2.78m/s^2的累积时间不小于2秒，并且加速度刷新时间	//
//	不超过100毫秒。																		//
//     SQ_GF::StatisticalQueue<int>* pQueue												//
//			= new SQ_GF::StatisticalQueue<int>(SQ_GF::QUEUE_FORM_TIME,3000,2000,100);	//
//     if 加速度值 > 2.78m/s^2															//
//			int result = pQueue->push(1);												//
//	   else																				//
//			int result = pQueue->push(0);												//
//     endif																			//
//     switch (result) {																//
//     case SQ_FLAG_MORE_THAN:															//
//			急刹车																		//
//     case SQ_FLAG_LESS_THAN:															//
//			正常行驶																	//
//	   }																				//
//																						//
//  2. 轻度疲劳统计。																	//
//	   条件：30秒内眨眼次数不小于20次。													//
//     SQ_GF::StatisticalQueue<int>* pQueue												//
//			= new SQ_GF::StatisticalQueue<int>(SQ_GF::QUEUE_FORM_TICK, 30000, 20);		//
//     if 眨眼																			//
//			int result = pQueue->push(1);												//
//	   else																				//
//			int result = pQueue->push(0);												//
//     endif																			//
//     switch (result) {																//
//     case SQ_FLAG_MORE_THAN:															//
//			轻度疲劳																	//
//     case SQ_FLAG_LESS_THAN:															//
//			正常																		//
//	   }																				//
//	   int cumulants = pQueue->getCumulants();	获取眨眼次数							//
//																						//
//  3. 行驶里程计算。																	//
//	   条件：1小时内车辆行驶里程，单位：m。												//
//     SQ_GF::StatisticalQueue<double>* pQueue											//
//			= new SQ_GF::StatisticalQueue<double>(SQ_GF::QUEUE_FORM_TIME, 3600000, 0);	//
//     pQueue->push(v);		其中 double v 表示车速，单位：m/ms							//
//	   里程 = pQueue->getCumulants();													//
//////////////////////////////////////////////////////////////////////////////////////////


#include <vector>
#include "../ThreadSafeQueue/ThreadSafeQueue.h"

#define SQ_FLAG_MORE_THAN	(1)
#define SQ_FLAG_LESS_THAN	(2)


#if defined(_WIN32)
#include <sys/timeb.h>
#include <Windows.h>
#else
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#endif


namespace SQ_GF
{
	typedef enum eQueueForm
	{
		QUEUE_FORM_TIME = 0,	// 周期内累积时间统计
		QUEUE_FORM_TICK = 1		// 周期内累积次数统计
	}QUEUE_FORM;

	template<class T>
	struct StatisticalElement
	{
		T data;				// 数据
		T cumulants;		// 累积值
		uint64_t timing;	// 有效时长
		uint64_t timestamp;	// 时间戳，单位：ms
	};

	template<class T>
	class StatisticalQueue
	{
	public:

		//////////////////////////////////////////////////////////////////////////
		//	StatisticalQueue(QUEUE_FORM, size_t, T, size_t)						//
		//	功能:																//
		//		构造函数。														//
		//	输入:																//
		//		nFormFlag		统计类型	SQ_FORM_TIME	|	SQ_FORM_TICK	//
		//		cycle			统计周期				单位：ms				//
		//		threshold		阈值		  单位：ms		|	  单位：次		//
		//		validPeriod		单次限制时间(0:不限制)	单位：ms				//
		//						只对SQ_FORM_TIME有效							//
		//////////////////////////////////////////////////////////////////////////

		StatisticalQueue(QUEUE_FORM nFormFlag, size_t cycle, T threshold, size_t validPeriod = 0, bool isGradient = false);


		void setParameter(size_t cycle, T threshold, size_t validPeriod);


		//////////////////////////////////////////////////////
		//	bool push(T)									//
		//	功能:											//
		//		输入数据判定累积值。						//
		//	返回值:											//
		//		判断结果。									//
		//			SQ_FLAG_MORE_THAN = 1	超过阈值		//
		//			SQ_FLAG_LESS_THAN = 2	不足阈值		//
		//	输入:											//
		//		data		数据							//
		//	输出:											//
		//		无。										//
		//////////////////////////////////////////////////////

		int push(T data);


		//////////////////////////////////////////////////////
		//	T getCumulants()								//
		//	功能:											//
		//		获取累积值。								//
		//	返回值:											//
		//		累积值，单位：ms。							//
		//	输入:											//
		//		无。										//
		//	输出:											//
		//		无。										//
		//////////////////////////////////////////////////////

		T getCumulants() { return m_nCumulants; }


		//////////////////////////////////////////////////////
		//	T getMean()										//
		//	功能:											//
		//		获取平均值。								//
		//	返回值:											//
		//		平均值。									//
		//	输入:											//
		//		无。										//
		//	输出:											//
		//		无。										//
		//////////////////////////////////////////////////////

		T getMean();

		T getMax();

		T getMin();


		//////////////////////////////////////////////////////
		//	uint64_t getTiming()							//
		//	功能:											//
		//		获取统计时长。								//
		//	返回值:											//
		//		统计时长，单位：ms。						//
		//	输入:											//
		//		无。										//
		//	输出:											//
		//		无。										//
		//////////////////////////////////////////////////////
		
		uint64_t getTiming(){ return m_nTiming; }


		//////////////////////////////////////////////////////
		//	uint64_t getContinueTiming()					//
		//	功能:											//
		//		获取连续触发时长。							//
		//	返回值:											//
		//		连续触发时长，单位：ms。					//
		//	输入:											//
		//		无。										//
		//	输出:											//
		//		无。										//
		//////////////////////////////////////////////////////

		uint64_t getContinueTiming() { return m_nContinueTiming; }


		size_t getCycle() { return m_nCycle; }
		size_t getValid() { return m_nValid; }
		T getThreshold() { return m_nThreshold; }


		// 获取毫秒级时间戳
		uint64_t getMillitm();


		void reset();

		// 获取斜率
		double getGradient()
		{
			if (m_bIsGradient) {
				double temp = m_vecData.size() * m_C - m_A * m_A;
				if (temp) {
					m_fGradient = (m_vecData.size() * m_D - m_A * m_B) / temp;
				}
				else {
					m_fGradient = 0;
				}
			}

			return m_fGradient;
		}

		double getVar()
		{
			size_t sz = m_vecData.size();
			if (sz == 0) {
				return 0.0;
			}

			double var = 0.0;
			T mean = this->getMean();
			for (size_t i = 0; i < sz; i++) {
				double diff = (double)(m_vecData[i].data - mean);
				var += (diff * diff);
			}
			var /= sz;

			return var;
		}


	private:

		// 数据缓存
		bool pushCache(T data);

		// 获取缓存队头元素 & 出队列
		bool popCache(StatisticalElement<T>& element);

		// 输入数据
		void pushData(StatisticalElement<T>& element);

		// 删除超时数据
		void popData();

		// 计算时间差：element1.timestamp - element2.timestamp
		uint64_t timeDiff(const StatisticalElement<T>& element1, const StatisticalElement<T>& element2);

		// 防止长时间不更新
		void update();

		// 防止更新过快内存爆满
		void suspend();


		QUEUE_FORM		m_eFormFlag;	// 队列类型
		size_t	m_nCycle;				// 统计周期，单位：ms
		size_t	m_nValid;				// 单帧限制时间
		T		m_nThreshold;			// 阈值
		T		m_nCumulants;			// 积分和
		uint64_t	m_nTiming;			// 统计时长
		uint64_t	m_nContinueTiming;	// 连续触发时间
		bool	m_bLock;				// 运行锁
		std::vector< StatisticalElement<T> >				m_vecData;		// 保存数据
		TSQ_GF::ThreadSafeQueue< StatisticalElement<T> >	m_queueCache;	// 缓存数据

		bool	m_bIsGradient;
		double	m_fGradient;	// 数据斜率（每秒的变化量）
		uint64_t m_nTick0;
		double	m_A;
		double	m_B;
		double	m_C;
		double	m_D;
	};
}


template<class T>
SQ_GF::StatisticalQueue<T>::StatisticalQueue(SQ_GF::QUEUE_FORM eFormFlag, size_t cycle, T threshold, size_t validPeriod, bool isGradient)
	: m_queueCache(100, TSQ_FORM_OVERFLOW_ACCEPT)
{
	m_eFormFlag = eFormFlag;
	m_nCycle = cycle;
	m_nValid = validPeriod;
	m_nThreshold = threshold;
	m_nCumulants = 0;
	m_nTiming = 0;
	m_nContinueTiming = 0;
	m_bLock = false;

	m_bIsGradient = isGradient;
	m_fGradient = 0;
	m_nTick0 = this->getMillitm();
	m_A = 0;
	m_B = 0;
	m_C = 0;
	m_D = 0;
}


template<class T>
void SQ_GF::StatisticalQueue<T>::setParameter(size_t cycle, T threshold, size_t validPeriod)
{
	m_nCycle = cycle;
	m_nThreshold = threshold;
	m_nValid = validPeriod;
}


//template<class T>
//void SQ_GF::StatisticalQueue<T>::update()
//{
//	size_t sz = m_vecData.size();
//	if (sz > 1) {
//		// 长时间不更新，则先更新一次
//		if (getMillitm() - m_vecData[sz - 1].timestamp >= m_nValid) {
//			this->push(m_vecData[sz - 1].data);
//		}
//	}
//}


template<class T>
T SQ_GF::StatisticalQueue<T>::getMean()
{
	if (m_nTiming == 0) {
		return (T)0;
	}
	else {
		switch (m_eFormFlag) {
		case QUEUE_FORM::QUEUE_FORM_TIME:
			return (T)(m_nCumulants / m_nTiming);
		case QUEUE_FORM::QUEUE_FORM_TICK:
			return (T)(m_nCumulants * m_nCycle / m_nTiming);
		default:
			return (T)0;
		}
	}
}


template<class T>
T SQ_GF::StatisticalQueue<T>::getMax()
{
	if (m_vecData.size() == 0) {
		return (T)0;
	}

	T valueMax = m_vecData[0].data;
	for (int i = 1; i < m_vecData.size(); i++) {
		if (valueMax < m_vecData[i].data) {
			valueMax = m_vecData[i].data;
		}
	}

	return valueMax;
}


template<class T>
T SQ_GF::StatisticalQueue<T>::getMin()
{
	if (m_vecData.size() == 0) {
		return (T)0;
	}

	T valueMin = m_vecData[0].data;
	for (int i = 1; i < m_vecData.size(); i++) {
		if (valueMin > m_vecData[i].data) {
			valueMin = m_vecData[i].data;
		}
	}

	return valueMin;
}


template<class T>
int SQ_GF::StatisticalQueue<T>::push(T data)
{
	// 输入缓存
	this->pushCache(data);

	// lock
	if (m_bLock) {
		return (m_nCumulants >= m_nThreshold) ? SQ_FLAG_MORE_THAN : SQ_FLAG_LESS_THAN;
	}
	else {
		m_bLock = true;
	}

	// 循环取值计算，输出最后一次的结果
	StatisticalElement<T> element;
	for (int i = 0; i < 10; i++) {
		// 取数据
		bool bResult = this->popCache(element);
		if (bResult) {
			// 录入数据
			this->pushData(element);
		}
		else {
			break;
		}
	}
		
	// 清除超时数据
	this->popData();
	// 根据阈值计算返回值
	int nResult = (m_nCumulants >= m_nThreshold) ? SQ_FLAG_MORE_THAN : SQ_FLAG_LESS_THAN;

	// unlock
	m_bLock = false;

	return nResult;
}


template<class T>
bool SQ_GF::StatisticalQueue<T>::pushCache(T data)
{
	StatisticalElement<T> element;
	element.timestamp = this->getMillitm();
	element.cumulants = 0;	// 缺省值
	element.timing = 0;	// 缺省值
	element.data = data;

	this->suspend();

	// 防止时间倒退
	int sz = m_vecData.size();
	if (sz > 0 && element.timestamp < m_vecData[sz - 1].timestamp) {
		this->reset();
	}

	return m_queueCache.push(element);
}


template<class T>
bool SQ_GF::StatisticalQueue<T>::popCache(StatisticalElement<T>& element)
{
	return m_queueCache.pop(element);
}


template<class T>
void SQ_GF::StatisticalQueue<T>::pushData(StatisticalElement<T>& element)
{
	size_t sz = m_vecData.size();
	if (sz == 0) {
		element.timing = 0;
		element.cumulants = 0;
		m_nTiming = 0;
		m_nContinueTiming = 0;
		m_nCumulants = 0;
	}
	else {
		// 有效累计时间
		uint64_t t = timeDiff(element, m_vecData[sz - 1]);
		if (t > m_nValid && m_nValid > 0) {
			t = m_nValid;
		}
		element.timing = t;
		m_nTiming += t;

		// 连续触发时长
		if (element.data && m_vecData[sz - 1].data) {
			m_nContinueTiming += t;
		}
		else {
			m_nContinueTiming = 0;
		}

		// 累积值
		switch (m_eFormFlag) {
		case QUEUE_FORM::QUEUE_FORM_TIME:
		{
			element.cumulants = element.data * t;
			m_nCumulants += element.cumulants;
		}
		break;
		case QUEUE_FORM::QUEUE_FORM_TICK:
			// if (element.data) {
			// 	element.cumulants = 1;
			// }
			// else {
			// 	element.cumulants = 0;
			// }
			element.cumulants = element.data;
			m_nCumulants += element.cumulants;
		}
	}

	// 录入数据
	m_vecData.push_back(element);

	// 梯度
	if (m_bIsGradient) {
		double x = (element.timestamp - m_nTick0) / 1000.0;
		m_A += x;
		m_B += element.data;
		m_C += x * x;
		m_D += x * element.data;

		//double temp = m_vecData.size() * m_C - m_A * m_A;
		//if (temp) {
		//	m_fGradient = (m_vecData.size() * m_D - m_A * m_B) / temp;
		//}
		//else {
		//	m_fGradient = 0;
		//}
	}
}


template<class T>
void SQ_GF::StatisticalQueue<T>::popData()
{
	size_t sz = m_vecData.size();
	if (sz < 2) {
		// 数据过短
		return;
	}

	// 二分法找边界
	size_t pos = 0;
	size_t posBegin = 0;
	size_t posEnd = sz - 1;
	while (posEnd - posBegin > 1) {
		uint64_t t = timeDiff(m_vecData[sz - 1], m_vecData[pos]);
		if (t > m_nCycle) {
			posBegin = pos;
		}
		else if (t < m_nCycle) {
			posEnd = pos;
		}
		else {
			posBegin = pos;
			posEnd = pos;
			break;
		}
		pos = (posBegin + posEnd) >> 1;
	}

	// 不超过统计时间
	if (timeDiff(m_vecData[sz - 1], m_vecData[pos]) > m_nCycle) {
		pos = posEnd;
	}
	// 更新累加值
	switch (m_eFormFlag) {
	case QUEUE_FORM::QUEUE_FORM_TIME:
		for (int i = 0; i < pos; i++) {
			m_nCumulants -= m_vecData[1].cumulants;
			m_nTiming -= m_vecData[1].timing;
			// 梯度
			if (m_bIsGradient) {
				double x = (m_vecData[1].timestamp - m_nTick0) / 1000.0;
				m_A -= x;
				m_B -= m_vecData[1].data;
				m_C -= x * x;
				m_D -= x * m_vecData[1].data;
			}
			m_vecData.erase(m_vecData.begin());
		}
		break;
	case QUEUE_FORM::QUEUE_FORM_TICK:
		for (int i = 0; i < pos; i++) {
			m_nCumulants -= m_vecData[1].cumulants;
			m_nTiming -= m_vecData[1].timing;
			// 梯度
			if (m_bIsGradient) {
				double x = (m_vecData[1].timestamp - m_nTick0) / 1000.0;
				m_A -= x;
				m_B -= m_vecData[1].data;
				m_C -= x * x;
				m_D -= x * m_vecData[1].data;
			}
			m_vecData.erase(m_vecData.begin());
		}
	}
}


template<class T>
uint64_t SQ_GF::StatisticalQueue<T>::timeDiff(const StatisticalElement<T>& element1, const StatisticalElement<T>& element2)
{
	return (element1.timestamp - element2.timestamp);
}


template<class T>
uint64_t SQ_GF::StatisticalQueue<T>::getMillitm()
{
#if defined(_WIN32)
	struct timeb tb;
	ftime(&tb);

	return (uint64_t)(tb.time * 1000 + tb.millitm);
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);

	return (uint64_t)(tv.tv_sec * 1000 + tv.tv_usec / 1000);
#endif
}


template<class T>
void SQ_GF::StatisticalQueue<T>::reset()
{
	while (true) {
		if (!m_bLock) {
			m_bLock = true;
			break;
		}
	}

	m_nCumulants = 0;
	m_nTiming = 0;

	m_vecData.clear();
	m_queueCache.clear();

	if (m_bIsGradient) {
		m_fGradient = 0;
		m_nTick0 = this->getMillitm();
		m_A = 0;
		m_B = 0;
		m_C = 0;
		m_D = 0;
	}

	m_bLock = false;
}


#define SQ_CONST_MIN_SUSPEND	(3)	// 最小输入间隔，单位：ms
template<class T>
void SQ_GF::StatisticalQueue<T>::suspend()
{
	size_t sz = m_vecData.size();
	if (sz > 0) {
		uint64_t timeDiff = this->getMillitm() - m_vecData[sz - 1].timestamp;
		if (timeDiff < SQ_CONST_MIN_SUSPEND) {
#ifdef _WIN32
			Sleep(SQ_CONST_MIN_SUSPEND - timeDiff);
#else
			usleep((SQ_CONST_MIN_SUSPEND - timeDiff) * 1000);
#endif
		}
	}
}