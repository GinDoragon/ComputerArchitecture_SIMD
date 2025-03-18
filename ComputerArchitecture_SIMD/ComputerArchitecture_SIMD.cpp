#include <immintrin.h>
#include <mmintrin.h>
#include <iostream>
#include <conio.h>
#include <xmmintrin.h>
#include <dvec.h>

static void mmx_operation() {
	__m64 a = _mm_set_pi8(100, 20, 30, 40, -50, 23, 12, -20);
	__m64 b = _mm_set_pi8(30, 10, -1, 1, -50, 50, -12, 20);
	__m64 res = _mm_adds_pi8(a, b);

	alignas(8) char result[8];
	memcpy(result, &res, sizeof(__m64));

	std::cout << "MMX Result: ";
	for (int i = 0; i < 8; i++)
	{
		std::cout << static_cast<int>(result[i]) << " ";
	}
	_mm_empty(); 
	std::cout << std::endl;
}

static void sse_operation() {
	__m128 a = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
	__m128 b = _mm_set_ps(1.0f, 2.0f, 2.5f, 3.5f);
	
	__m128 cmp = _mm_cmpeq_ps(a, b);
	__m128 hadd = _mm_hadd_ps(a, b);

	alignas(16) int cmp_result[4];
	_mm_store_si128(reinterpret_cast<__m128i*>(cmp_result), _mm_castps_si128(cmp));

	std::cout << "SSE Comparison Result: ";
	for (auto val : cmp_result)
	{
		std::cout << val << " ";
	}
	std::cout << std::endl;

	alignas(16) float hadd_result[4];
	_mm_store_ps(hadd_result, hadd);

	std::cout << "SSE Hadd Result: ";
	for (auto val : hadd_result)
	{
		std::cout << val << " ";
	}
	std::cout << std::endl;
}

static void avx_operation() {
	__m256 vec = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
	
	vec = _mm256_permute_ps(vec, 0b00011011);

	alignas(32) float permute_result[8];
	_mm256_store_ps(permute_result, vec);

	std::cout << "AVX Permute Result: ";
	for (int i = 0; i < 8; ++i)
	{
		std::cout << permute_result[i] << " ";
	}
	std::cout << std::endl;
}

static void dvec_operations() {
	F64vec2 vec2_1(0.1, 1.0);
	F64vec2 vec2_2(95.1, -2.1);
	F64vec2 vec2_res = vec2_1 + vec2_2;

	std::cout << "\nF64vec2 | Vec SUM: ";
	for (int i = 1; i >= 0; i--)
	{
		std::cout << vec2_res[i] << " ";
	}
	F64vec4 vec4_1(0.1, 1.0, 0.5, 0.6);
	F64vec4 vec4_2(95.1, -2.1, 0.4, 0.3);
	F64vec4 vec4_res = vec4_1 + vec4_2;
	std::cout << "\nF64vec4 | Vec SUM: ";
	for (int i = 3; i >= 0; i--)
	{
		std::cout << vec4_res[i] << " ";
	}
	std::cout << std::endl;
}

int main()
{
	mmx_operation();
	sse_operation();
	avx_operation();
	dvec_operations();
	return 0;
}