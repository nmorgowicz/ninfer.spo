add_executable(ninfer-spo
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer-spo)
target_link_libraries(ninfer-spo PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-spo-serve serve/main.cpp)
ninfer_internal_includes(ninfer-spo-serve)
target_link_libraries(ninfer-spo-serve PRIVATE ninfer_serve ninfer_product_logging)

add_executable(ninfer-spo-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp
  perplexity/kl.cpp)
ninfer_internal_includes(ninfer-spo-perplexity)
target_link_libraries(ninfer-spo-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)

# Offline calibration is opt-in; this target is not part of the default app build.
add_executable(ninfer-spo-gemma4-calibrate EXCLUDE_FROM_ALL gemma4_calibrate.cu)
ninfer_internal_includes(ninfer-spo-gemma4-calibrate)
target_link_libraries(ninfer-spo-gemma4-calibrate PRIVATE ninfer_model_runtime ninfer::json)
