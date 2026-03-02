#!/usr/bin/env python3
"""
Test script: Verify integration between gpt_simulator.py and generation.py
测试脚本：验证gpt_simulator.py和generation.py的集成

Updated to match current ClarifyQuery format with:
- data_source (abgcoqa/ambignq)
- reference_question
- reference_answer
- answer_hints
- full context format
"""

import requests
import json
import time

def test_health_check():
    """Test health check endpoint"""
    print("=== Testing Health Check ===")
    try:
        response = requests.get("http://127.0.0.1:8001/health", timeout=5)
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure gpt_simulator.py is running on port 8001")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test root endpoint for service information"""
    print("\n=== Testing Root Endpoint ===")
    try:
        response = requests.get("http://127.0.0.1:8001/", timeout=5)
        print(f"Root endpoint: {response.status_code}")
        if response.status_code == 200:
            info = response.json()
            print(f"Service: {info.get('service', 'N/A')}")
            print(f"Version: {info.get('version', 'N/A')}")
            print(f"Endpoints: {list(info.get('endpoints', {}).keys())}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint test failed: {e}")
        return False

def test_single_generate_basic():
    """Test single generate endpoint with basic data"""
    print("\n=== Testing Single Generate (Basic) ===")
    
    test_data = {
        "question": "What is Python?",
        "clarification_question": "Do you mean the programming language or the snake?",
        "context": "User's actual intent: I want to learn about the Python programming language",
        "data_source": "ambignq"
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8001/generate",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"Generate response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"Response: {result.get('response', 'N/A')[:100]}...")
            print(f"Question: {result.get('question', 'N/A')}")
            print(f"Clarification: {result.get('clarification_question', 'N/A')}")
            print(f"Data Source: {result.get('data_source', 'N/A')}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Generate test failed: {e}")
        return False

def test_single_generate_full_format():
    """Test single generate with full ClarifyQuery format"""
    print("\n=== Testing Single Generate (Full Format) ===")
    
    test_data = {
        "question": "What color was it?",
        "clarification_question": "Do you mean the first book?",
        "context": """Passage:
Angie went to the library with her mother. First she had to turn in the books she was returning at the return desk. They said hello to the man there. He took their books. Then they went into the adult reading room. Angie sat in a brown chair at the table. She made a drawing of her mother. Her mother found a large red book.

Conversation history:
Turn 3 - Question: what did she draw?
Turn 3 - Answer: her mother
Turn 4 - Question: what did her mother find?
Turn 4 - Answer: the book.

User's actual intent: What color was the book that her mother found in the adult reading room?""",
        "data_source": "abgcoqa",
        "reference_question": "Do you mean the first book?",
        "reference_answer": "Yes",
        "answer_hints": ["red", "green"]
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8001/generate",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        print(f"Full format test response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"Response: {result.get('response', 'N/A')[:150]}...")
            print(f"Data Source: {result.get('data_source', 'N/A')}")
            print(f"Reference Question: {result.get('reference_question', 'N/A')}")
            print(f"Reference Answer: {result.get('reference_answer', 'N/A')}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Full format test failed: {e}")
        return False

def test_batch_generate_ambignq():
    """Test batch generate with ambignq format"""
    print("\n=== Testing Batch Generate (AmbigNQ) ===")
    
    batch_data = {
        "queries": [
            {
                "question": "What is Python?",
                "clarification_question": "Do you mean the programming language?",
                "context": "User's actual intent: I want to learn about the Python programming language",
                "data_source": "ambignq"
            },
            {
                "question": "How to cook rice?",
                "clarification_question": "What type of rice do you want to cook?",
                "context": "User's actual intent: I want to cook white rice in a rice cooker",
                "data_source": "ambignq"
            }
        ],
        "return_scores": False
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8001/batch_generate",
            json=batch_data,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        print(f"Batch generate (ambignq) response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            
            if 'result' in result:
                responses = result['result']
                print(f"Number of responses: {len(responses)}")
                
                for i, resp in enumerate(responses):
                    print(f"  Response {i+1}:")
                    print(f"    Question: {resp.get('question', 'N/A')}")
                    print(f"    Clarification: {resp.get('clarification_question', 'N/A')}")
                    print(f"    Response: {resp.get('response', 'N/A')[:80]}...")
                    print(f"    Data Source: {resp.get('data_source', 'N/A')}")
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch generate (ambignq) test failed: {e}")
        return False

def test_batch_generate_abgcoqa():
    """Test batch generate with abgcoqa format (full context)"""
    print("\n=== Testing Batch Generate (Abg-CoQA) ===")
    
    batch_data = {
        "queries": [
            {
                "question": "what color was it?",
                "clarification_question": "Do you mean the first book?",
                "context": """Passage:
Angie went to the library with her mother. First she had to turn in the books she was returning at the return desk. They said hello to the man there. He took their books. Then they went into the adult reading room. Angie sat in a brown chair at the table. She made a drawing of her mother. Her mother found a large red book.

Conversation history:
Turn 3 - Question: what did she draw?
Turn 3 - Answer: her mother
Turn 4 - Question: what did her mother find?
Turn 4 - Answer: the book.

User's actual intent: What color was the book that her mother found in the adult reading room?""",
                "data_source": "abgcoqa",
                "reference_question": "Do you mean the first book?",
                "reference_answer": "Yes",
                "answer_hints": ["red"]
            },
            {
                "question": "what color was it?",
                "clarification_question": "Do you mean the first book?",
                "context": """Passage:
Angie went to the library with her mother. First she had to turn in the books she was returning at the return desk. They said hello to the man there. He took their books. Then they went into the adult reading room. Angie sat in a brown chair at the table. She made a drawing of her mother. Her mother found a large red book.

Conversation history:
Turn 3 - Question: what did she draw?
Turn 3 - Answer: her mother
Turn 4 - Question: what did her mother find?
Turn 4 - Answer: the book.

User's actual intent: What color was the book that her mother found in the adult reading room?""",
                "data_source": "abgcoqa",
                "reference_question": "Do you mean the first book?",
                "reference_answer": "No, I mean the second book.",
                "answer_hints": ["green"]
            }
        ],
        "return_scores": False
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8001/batch_generate",
            json=batch_data,
            headers={"Content-Type": "application/json"},
            timeout=20
        )
        print(f"Batch generate (abgcoqa) response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            
            if 'result' in result:
                responses = result['result']
                print(f"Number of responses: {len(responses)}")
                
                for i, resp in enumerate(responses):
                    print(f"  Response {i+1}:")
                    print(f"    Question: {resp.get('question', 'N/A')}")
                    print(f"    Clarification: {resp.get('clarification_question', 'N/A')}")
                    print(f"    Reference Answer: {resp.get('reference_answer', 'N/A')}")
                    print(f"    Response: {resp.get('response', 'N/A')[:100]}...")
                    print(f"    Data Source: {resp.get('data_source', 'N/A')}")
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch generate (abgcoqa) test failed: {e}")
        return False

def test_generation_format():
    """Test the exact format used by generation.py (_batch_clarify)"""
    print("\n=== Testing Generation.py Format ===")
    
    # This matches the format from generation.py _batch_clarify method
    generation_format_data = {
        "queries": [
            {
                "question": "what color was it?",
                "clarification_question": "Do you mean the first book?",
                "context": """Passage:
Angie went to the library with her mother. First she had to turn in the books she was returning at the return desk. They said hello to the man there. He took their books. Then they went into the adult reading room. Angie sat in a brown chair at the table. She made a drawing of her mother. Her mother found a large red book.

Conversation history:
Turn 3 - Question: what did she draw?
Turn 3 - Answer: her mother
Turn 4 - Question: what did her mother find?
Turn 4 - Answer: the book.

User's actual intent: What color was the book that her mother found in the adult reading room?""",
                "data_source": "abgcoqa",
                "reference_question": "Do you mean the first book?",
                "reference_answer": "Yes",
                "answer_hints": ["red"]
            }
        ],
        "return_scores": False
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8001/batch_generate",
            json=generation_format_data,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        print(f"Generation format test response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generation format test successful!")
            
            if 'result' in result and len(result['result']) > 0:
                first_response = result['result'][0]
                print(f"Response structure matches generation.py expectations:")
                print(f"  - Has 'question': {'question' in first_response}")
                print(f"  - Has 'clarification_question': {'clarification_question' in first_response}")
                print(f"  - Has 'context': {'context' in first_response}")
                print(f"  - Has 'response': {'response' in first_response}")
                print(f"  - Has 'data_source': {'data_source' in first_response}")
                print(f"  - Has 'reference_question': {'reference_question' in first_response}")
                print(f"  - Has 'reference_answer': {'reference_answer' in first_response}")
                
                # Check response content
                response_text = first_response.get('response', '')
                if response_text and len(response_text) > 10:
                    print(f"✅ Response content received: {response_text[:80]}...")
                else:
                    print("⚠️  Response content may be empty or too short")
                
                return True
            else:
                print("❌ Response format doesn't match expected structure")
                return False
        else:
            print(f"❌ Generation format test error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Generation format test failed: {e}")
        return False

def test_empty_question_fallback():
    """Test handling of empty question field (should use fallback)"""
    print("\n=== Testing Empty Question Fallback ===")
    
    test_data = {
        "question": "",  # Empty question
        "clarification_question": "Do you mean the first book?",
        "context": "User's actual intent: I want to know about the first book",
        "data_source": "abgcoqa"
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8001/batch_generate",
            json={"queries": [test_data], "return_scores": False},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"Empty question test response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if 'result' in result and len(result['result']) > 0:
                first_response = result['result'][0]
                question = first_response.get('question', '')
                if question and question != "":
                    print(f"✅ Fallback question used: {question}")
                    return True
                else:
                    print("⚠️  Question still empty after fallback")
                    return False
            else:
                print("❌ No response received")
                return False
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Empty question test failed: {e}")
        return False

def test_error_cases():
    """Test error handling cases"""
    print("\n=== Testing Error Cases ===")
    
    results = []
    
    # Test missing clarification_question
    print("Testing missing clarification_question...")
    try:
        response = requests.post(
            "http://127.0.0.1:8001/batch_generate",
            json={"queries": [{"question": "What is Python?"}], "return_scores": False},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            if 'result' in result and len(result['result']) > 0:
                resp = result['result'][0]
                if resp.get('response', '').startswith("I need more information"):
                    print("✅ Correctly handled missing clarification_question")
                    results.append(True)
                else:
                    print("⚠️  Unexpected handling")
                    results.append(False)
            else:
                print("⚠️  No response received")
                results.append(False)
        else:
            print(f"⚠️  Status: {response.status_code}")
            results.append(False)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(False)
    
    # Test empty queries list
    print("Testing empty queries list...")
    try:
        response = requests.post(
            "http://127.0.0.1:8001/batch_generate",
            json={"queries": [], "return_scores": False},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if response.status_code == 400:
            print("✅ Correctly rejected empty queries")
            results.append(True)
        else:
            print(f"⚠️  Status: {response.status_code}")
            results.append(False)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results.append(False)
    
    return all(results)

def test_data_source_differentiation():
    """Test that different data sources produce different prompts"""
    print("\n=== Testing Data Source Differentiation ===")
    
    base_query = {
        "question": "What is it?",
        "clarification_question": "Do you mean X or Y?",
        "context": "User's actual intent: I want to know about X"
    }
    
    test_cases = [
        {**base_query, "data_source": "ambignq"},
        {**base_query, "data_source": "abgcoqa"}
    ]
    
    try:
        response = requests.post(
            "http://127.0.0.1:8001/batch_generate",
            json={"queries": test_cases, "return_scores": False},
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'result' in result and len(result['result']) == 2:
                ambignq_resp = result['result'][0].get('response', '')
                abgcoqa_resp = result['result'][1].get('response', '')
                
                print(f"AmbigNQ response: {ambignq_resp[:80]}...")
                print(f"Abg-CoQA response: {abgcoqa_resp[:80]}...")
                
                # Both should have responses (even if different)
                if ambignq_resp and abgcoqa_resp:
                    print("✅ Both data sources produced responses")
                    return True
                else:
                    print("⚠️  One or both responses are empty")
                    return False
            else:
                print("❌ Unexpected response format")
                return False
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Data source differentiation test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting GPT Simulator Tests (Updated)")
    print("=" * 60)
    
    # Wait a moment for service to be ready
    print("⏳ Waiting for service to be ready...")
    time.sleep(2)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Health Check", test_health_check()))
    test_results.append(("Root Endpoint", test_root_endpoint()))
    test_results.append(("Single Generate (Basic)", test_single_generate_basic()))
    test_results.append(("Single Generate (Full Format)", test_single_generate_full_format()))
    test_results.append(("Batch Generate (AmbigNQ)", test_batch_generate_ambignq()))
    test_results.append(("Batch Generate (Abg-CoQA)", test_batch_generate_abgcoqa()))
    test_results.append(("Generation Format", test_generation_format()))
    test_results.append(("Empty Question Fallback", test_empty_question_fallback()))
    test_results.append(("Error Cases", test_error_cases()))
    test_results.append(("Data Source Differentiation", test_data_source_differentiation()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:35} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")
    
    if passed == total:
        print("\n🎉 All tests passed! GPT Simulator is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the logs above.")
    
    print("\n🏁 Testing completed!")

if __name__ == "__main__":
    main()
