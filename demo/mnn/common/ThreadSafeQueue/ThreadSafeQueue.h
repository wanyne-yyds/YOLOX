#pragma once
#define TSQ_VERSION		(V1.0.0)

#include <queue>
#include <mutex>

#define TSQ_FORM_OVERFLOW_REJECT	(1)		// ����ܾ�
#define TSQ_FORM_OVERFLOW_ACCEPT	(2)		// �������,ɾ����ͷ

namespace TSQ_GF
{
	template<class T>
	class ThreadSafeQueue
	{
	public:
		ThreadSafeQueue(size_t maxSize, int nFormFlag = TSQ_FORM_OVERFLOW_REJECT);
		~ThreadSafeQueue();

		// ��ĩβ����һ��Ԫ��
		bool push(const T& element);

		// ��ȡ��ͷԪ�� & ɾ��
		bool pop(T& element);

		// ���ض�����Ԫ�صĸ���
		size_t size();

		// �������
		void clear();


	private:

		int m_nFormFlag;		// ��������
		size_t m_nMaxSize;
		std::queue<T> m_queue;
		std::mutex* m_mutex;
	};
}


template<class T>
TSQ_GF::ThreadSafeQueue<T>::ThreadSafeQueue(size_t maxSize, int nFormFlag)
{
	switch (nFormFlag) {
	case TSQ_FORM_OVERFLOW_REJECT:
	case TSQ_FORM_OVERFLOW_ACCEPT:
		m_nFormFlag = nFormFlag;
		break;
	default:
		nFormFlag = TSQ_FORM_OVERFLOW_REJECT;
	}

	m_nMaxSize = maxSize;

	m_mutex = new std::mutex();
}


template<class T>
TSQ_GF::ThreadSafeQueue<T>::~ThreadSafeQueue()
{
	this->clear();

	if (m_mutex) {
		delete m_mutex;
		m_mutex = 0;
	}
}


template<class T>
void TSQ_GF::ThreadSafeQueue<T>::clear()
{
	m_mutex->lock();

	while (!m_queue.empty()) {
		m_queue.pop();
	}

	m_mutex->unlock();
}


template<class T>
bool TSQ_GF::ThreadSafeQueue<T>::push(const T& element)
{
	// full queue
	if (m_queue.size() >= m_nMaxSize) {
		switch (m_nFormFlag) {
		case TSQ_FORM_OVERFLOW_REJECT:
			return false;
		case TSQ_FORM_OVERFLOW_ACCEPT:
		default:
			;
		}
	}

	// lock
	m_mutex->lock();

	// push
	while (m_queue.size() >= m_nMaxSize) {
		m_queue.pop();
	}
	m_queue.push(element);

	// unlock
	m_mutex->unlock();

	return true;
}


template<class T>
bool TSQ_GF::ThreadSafeQueue<T>::pop(T& element)
{
	// empty
	if (!this->size()) {
		return false;
	}

	// lock
	m_mutex->lock();

	element = m_queue.front();

	m_queue.pop();

	// unlock
	m_mutex->unlock();

	return true;
}


template<class T>
size_t TSQ_GF::ThreadSafeQueue<T>::size()
{
	// lock
	m_mutex->lock();

	size_t sz = m_queue.size();

	// unlock
	m_mutex->unlock();

	return sz;
}