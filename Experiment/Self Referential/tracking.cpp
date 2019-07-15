/*

#include <iostream>
#include <limits>
#include <memory>
#include <vector>

template<class T>
class TrackingAllocator
{
public:
	using value_type = T;

	using pointer = T *;
	using const_pointer = const T*;

	using size_type = size_t;

	TrackingAllocator() = default;

	template<class U>
	TrackingAllocator(const TrackingAllocator<U>& other) {}

	~TrackingAllocator() = default;

	pointer allocate(size_type numObjects)
	{
		mAllocations += numObjects;
		return static_cast<pointer>(operator new(sizeof(T) * numObjects));
	}

	void deallocate(pointer p, size_type numObjects)
	{
		operator delete(p);
	}

	size_type get_allocations() const
	{
		return mAllocations;
	}

private:
	static size_type mAllocations;
};

template<class T>
typename TrackingAllocator<T>::size_type TrackingAllocator<T>::mAllocations = 0;

int main()
{
	std::vector<int, TrackingAllocator<int>> v(5);
	std::cout << v.get_allocator().get_allocations() << std::endl;

	return 0;
}*/