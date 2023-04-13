#pragma once
#define SQ_VERSION		(V1.0.13_20221115)

//////////////////////////////////////////////////////////////////////////////////////////
//							class StatisticalQueue ʹ��˵��								//
//      ��������ͳ�ƹ̶�ʱ�������ڵ�֡״̬��ʱ���Ȩ�ۻ�ֵ������ۻ�ֵ��				//
//	    int push(T data); ����״̬��Ȩ�أ������ۻ�״̬��								//
//	    T getCumulants(); �����ۻ�ֵ��													//
//	    ͳ�Ƶ�ʱ�����ڲ�ȡ�ͽ�ԭ�򣺼���cycle=100ms��ǰ����֡��ʱ���ֱ���96ms��102ms��	//
//	��ôͳ������ѡ��102ms��																//
//	    validPeriod�ǵ�֡ʱ�����Ʋ����������ǵ�ͳ�ƹ����£���һ֡��״̬�����ǰ����֡	//
//	֮���״̬������һЩ����Ԥ�ϵ�ԭ����ǰ��֡��ʱ������������ʱ��Ĵ���ͳ�ƣ�	//
//	�����󱨡����ǰ����֡ʱ��������validPeriod����ʹ��validPeriod���ʱ������Ĭ	//
//	��ֵ0��ʾ�����ơ���Ȼ�ò���ֻ��SQ_FORM_TIME��Ч������ͳ�Ƹ���֡ʱ���޹ء�			//
//																						//
//  ʹ�ð�����																			//
//  1. �������ж���																		//
//	   ������3���ڳ����ƶ����ٶȴ���2.78m/s^2���ۻ�ʱ�䲻С��2�룬���Ҽ��ٶ�ˢ��ʱ��	//
//	������100���롣																		//
//     SQ_GF::StatisticalQueue<int>* pQueue												//
//			= new SQ_GF::StatisticalQueue<int>(SQ_GF::QUEUE_FORM_TIME,3000,2000,100);	//
//     if ���ٶ�ֵ > 2.78m/s^2															//
//			int result = pQueue->push(1);												//
//	   else																				//
//			int result = pQueue->push(0);												//
//     endif																			//
//     switch (result) {																//
//     case SQ_FLAG_MORE_THAN:															//
//			��ɲ��																		//
//     case SQ_FLAG_LESS_THAN:															//
//			������ʻ																	//
//	   }																				//
//																						//
//  2. ���ƣ��ͳ�ơ�																	//
//	   ������30����գ�۴�����С��20�Ρ�													//
//     SQ_GF::StatisticalQueue<int>* pQueue												//
//			= new SQ_GF::StatisticalQueue<int>(SQ_GF::QUEUE_FORM_TICK, 30000, 20);		//
//     if գ��																			//
//			int result = pQueue->push(1);												//
//	   else																				//
//			int result = pQueue->push(0);												//
//     endif																			//
//     switch (result) {																//
//     case SQ_FLAG_MORE_THAN:															//
//			���ƣ��																	//
//     case SQ_FLAG_LESS_THAN:															//
//			����																		//
//	   }																				//
//	   int cumulants = pQueue->getCumulants();	��ȡգ�۴���							//
//																						//
//  3. ��ʻ��̼��㡣																	//
//	   ������1Сʱ�ڳ�����ʻ��̣���λ��m��												//
//     SQ_GF::StatisticalQueue<double>* pQueue											//
//			= new SQ_GF::StatisticalQueue<double>(SQ_GF::QUEUE_FORM_TIME, 3600000, 0);	//
//     pQueue->push(v);		���� double v ��ʾ���٣���λ��m/ms							//
//	   ��� = pQueue->getCumulants();													//
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
		QUEUE_FORM_TIME = 0,	// �������ۻ�ʱ��ͳ��
		QUEUE_FORM_TICK = 1		// �������ۻ�����ͳ��
	}QUEUE_FORM;

	template<class T>
	struct StatisticalElement
	{
		T data;				// ����
		T cumulants;		// �ۻ�ֵ
		uint64_t timing;	// ��Чʱ��
		uint64_t timestamp;	// ʱ�������λ��ms
	};

	template<class T>
	class StatisticalQueue
	{
	public:

		//////////////////////////////////////////////////////////////////////////
		//	StatisticalQueue(QUEUE_FORM, size_t, T, size_t)						//
		//	����:																//
		//		���캯����														//
		//	����:																//
		//		nFormFlag		ͳ������	SQ_FORM_TIME	|	SQ_FORM_TICK	//
		//		cycle			ͳ������				��λ��ms				//
		//		threshold		��ֵ		  ��λ��ms		|	  ��λ����		//
		//		validPeriod		��������ʱ��(0:������)	��λ��ms				//
		//						ֻ��SQ_FORM_TIME��Ч							//
		//////////////////////////////////////////////////////////////////////////

		StatisticalQueue(QUEUE_FORM nFormFlag, size_t cycle, T threshold, size_t validPeriod = 0, bool isGradient = false);


		void setParameter(size_t cycle, T threshold, size_t validPeriod);


		//////////////////////////////////////////////////////
		//	bool push(T)									//
		//	����:											//
		//		���������ж��ۻ�ֵ��						//
		//	����ֵ:											//
		//		�жϽ����									//
		//			SQ_FLAG_MORE_THAN = 1	������ֵ		//
		//			SQ_FLAG_LESS_THAN = 2	������ֵ		//
		//	����:											//
		//		data		����							//
		//	���:											//
		//		�ޡ�										//
		//////////////////////////////////////////////////////

		int push(T data);


		//////////////////////////////////////////////////////
		//	T getCumulants()								//
		//	����:											//
		//		��ȡ�ۻ�ֵ��								//
		//	����ֵ:											//
		//		�ۻ�ֵ����λ��ms��							//
		//	����:											//
		//		�ޡ�										//
		//	���:											//
		//		�ޡ�										//
		//////////////////////////////////////////////////////

		T getCumulants() { return m_nCumulants; }


		//////////////////////////////////////////////////////
		//	T getMean()										//
		//	����:											//
		//		��ȡƽ��ֵ��								//
		//	����ֵ:											//
		//		ƽ��ֵ��									//
		//	����:											//
		//		�ޡ�										//
		//	���:											//
		//		�ޡ�										//
		//////////////////////////////////////////////////////

		T getMean();

		T getMax();

		T getMin();


		//////////////////////////////////////////////////////
		//	uint64_t getTiming()							//
		//	����:											//
		//		��ȡͳ��ʱ����								//
		//	����ֵ:											//
		//		ͳ��ʱ������λ��ms��						//
		//	����:											//
		//		�ޡ�										//
		//	���:											//
		//		�ޡ�										//
		//////////////////////////////////////////////////////
		
		uint64_t getTiming(){ return m_nTiming; }


		//////////////////////////////////////////////////////
		//	uint64_t getContinueTiming()					//
		//	����:											//
		//		��ȡ��������ʱ����							//
		//	����ֵ:											//
		//		��������ʱ������λ��ms��					//
		//	����:											//
		//		�ޡ�										//
		//	���:											//
		//		�ޡ�										//
		//////////////////////////////////////////////////////

		uint64_t getContinueTiming() { return m_nContinueTiming; }


		size_t getCycle() { return m_nCycle; }
		size_t getValid() { return m_nValid; }
		T getThreshold() { return m_nThreshold; }


		// ��ȡ���뼶ʱ���
		uint64_t getMillitm();


		void reset();

		// ��ȡб��
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

		// ���ݻ���
		bool pushCache(T data);

		// ��ȡ�����ͷԪ�� & ������
		bool popCache(StatisticalElement<T>& element);

		// ��������
		void pushData(StatisticalElement<T>& element);

		// ɾ����ʱ����
		void popData();

		// ����ʱ��element1.timestamp - element2.timestamp
		uint64_t timeDiff(const StatisticalElement<T>& element1, const StatisticalElement<T>& element2);

		// ��ֹ��ʱ�䲻����
		void update();

		// ��ֹ���¹����ڴ汬��
		void suspend();


		QUEUE_FORM		m_eFormFlag;	// ��������
		size_t	m_nCycle;				// ͳ�����ڣ���λ��ms
		size_t	m_nValid;				// ��֡����ʱ��
		T		m_nThreshold;			// ��ֵ
		T		m_nCumulants;			// ���ֺ�
		uint64_t	m_nTiming;			// ͳ��ʱ��
		uint64_t	m_nContinueTiming;	// ��������ʱ��
		bool	m_bLock;				// ������
		std::vector< StatisticalElement<T> >				m_vecData;		// ��������
		TSQ_GF::ThreadSafeQueue< StatisticalElement<T> >	m_queueCache;	// ��������

		bool	m_bIsGradient;
		double	m_fGradient;	// ����б�ʣ�ÿ��ı仯����
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
//		// ��ʱ�䲻���£����ȸ���һ��
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
	// ���뻺��
	this->pushCache(data);

	// lock
	if (m_bLock) {
		return (m_nCumulants >= m_nThreshold) ? SQ_FLAG_MORE_THAN : SQ_FLAG_LESS_THAN;
	}
	else {
		m_bLock = true;
	}

	// ѭ��ȡֵ���㣬������һ�εĽ��
	StatisticalElement<T> element;
	for (int i = 0; i < 10; i++) {
		// ȡ����
		bool bResult = this->popCache(element);
		if (bResult) {
			// ¼������
			this->pushData(element);
		}
		else {
			break;
		}
	}
		
	// �����ʱ����
	this->popData();
	// ������ֵ���㷵��ֵ
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
	element.cumulants = 0;	// ȱʡֵ
	element.timing = 0;	// ȱʡֵ
	element.data = data;

	this->suspend();

	// ��ֹʱ�䵹��
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
		// ��Ч�ۼ�ʱ��
		uint64_t t = timeDiff(element, m_vecData[sz - 1]);
		if (t > m_nValid && m_nValid > 0) {
			t = m_nValid;
		}
		element.timing = t;
		m_nTiming += t;

		// ��������ʱ��
		if (element.data && m_vecData[sz - 1].data) {
			m_nContinueTiming += t;
		}
		else {
			m_nContinueTiming = 0;
		}

		// �ۻ�ֵ
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

	// ¼������
	m_vecData.push_back(element);

	// �ݶ�
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
		// ���ݹ���
		return;
	}

	// ���ַ��ұ߽�
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

	// ������ͳ��ʱ��
	if (timeDiff(m_vecData[sz - 1], m_vecData[pos]) > m_nCycle) {
		pos = posEnd;
	}
	// �����ۼ�ֵ
	switch (m_eFormFlag) {
	case QUEUE_FORM::QUEUE_FORM_TIME:
		for (int i = 0; i < pos; i++) {
			m_nCumulants -= m_vecData[1].cumulants;
			m_nTiming -= m_vecData[1].timing;
			// �ݶ�
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
			// �ݶ�
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


#define SQ_CONST_MIN_SUSPEND	(3)	// ��С����������λ��ms
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