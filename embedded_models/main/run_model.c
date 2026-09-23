#include <stdio.h>
#include <inttypes.h>
#include <limits.h>

#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_chip_info.h"
#include "esp_system.h"
#include "esp_timer.h"

#include "random_forest_model.h"
#include "test_data_rf.h"

#define NUM_REPETITIONS 10

void app_main(void)
{
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);

    printf("Chip: %s\n", CONFIG_IDF_TARGET);
    printf("CPU cores: %d\n", chip_info.cores);

    printf("Features: %s%s%s%s\n",
           (chip_info.features & CHIP_FEATURE_WIFI_BGN) ? "WiFi/" : "",
           (chip_info.features & CHIP_FEATURE_BT) ? "BT/" : "",
           (chip_info.features & CHIP_FEATURE_BLE) ? "BLE/" : "",
           (chip_info.features & CHIP_FEATURE_IEEE802154)
               ? "802.15.4 (Zigbee/Thread)"
               : "");

    unsigned major_rev = chip_info.revision / 100;
    unsigned minor_rev = chip_info.revision % 100;

    printf("Silicon revision: v%d.%d\n", major_rev, minor_rev);

    printf("\n");
    printf("Random Forest configuration:\n");
    printf("  Features: %d\n", NUM_FEATURES);
    printf("  Test samples: %d\n", NUM_SAMPLES);
    printf("  Trees: 100\n");
    printf("  Repetitions: %d\n", NUM_REPETITIONS);

     // Store heap information before inference.
    uint32_t free_heap_before = esp_get_free_heap_size();
    uint32_t min_heap_before = esp_get_minimum_free_heap_size();

    printf("\n");
    printf("Free heap before inference: %" PRIu32 " bytes\n",
           free_heap_before);

    printf("\n");
    printf("Starting inference...\n");
    printf("----------------------------------------\n");

    /*
     * Warm-up inference.
     *
     * This is not included in the benchmark.
     */
    volatile int32_t prediction = random_forest_predict(
        test_data[0],
        NUM_FEATURES
    );

    int64_t total_time = 0;
    int64_t min_time = INT64_MAX;
    int64_t max_time = 0;

    int total_inferences = NUM_SAMPLES * NUM_REPETITIONS;

    for (int repeat = 0; repeat < NUM_REPETITIONS; repeat++) {

        for (int i = 0; i < NUM_SAMPLES; i++) {

            int64_t start_time = esp_timer_get_time();

            prediction = random_forest_predict(
                test_data[i],
                NUM_FEATURES
            );

            int64_t elapsed = esp_timer_get_time() - start_time;

            total_time += elapsed;

            if (elapsed < min_time) {
                min_time = elapsed;
            }

            if (elapsed > max_time) {
                max_time = elapsed;
            }
        }
    }
     // Heap information after inference.

    uint32_t free_heap_after = esp_get_free_heap_size();
    uint32_t min_heap_after = esp_get_minimum_free_heap_size();
    
    printf("\n");
    printf("Example predictions\n");
    printf("--------------------\n");

    for (int i = 0; i < 10 && i < NUM_SAMPLES; i++) {

        int32_t p = random_forest_predict(
            test_data[i],
            NUM_FEATURES
        );

        printf(
            "Sample %d -> prediction: %" PRId32 "\n",
            i,
            p
        );
    }

    printf("\n");
    printf("Inference benchmark\n");
    printf("--------------------\n");

    printf("Samples:           %d\n", NUM_SAMPLES);
    printf("Repetitions:       %d\n", NUM_REPETITIONS);
    printf("Total inferences:  %d\n", total_inferences);

    printf("Total time:        %" PRId64 " us\n",
           total_time);

    printf("Min:               %" PRId64 " us\n",
           min_time);

    printf("Max:               %" PRId64 " us\n",
           max_time);

    printf("Average:           %" PRId64 " us\n",
           total_time / total_inferences);

    printf("Last prediction:   %" PRId32 "\n",
           prediction);

    printf("\n");
    printf("Memory\n");
    printf("--------------------\n");

    printf("Free heap before:  %" PRIu32 " bytes\n",
           free_heap_before);

    printf("Free heap after:   %" PRIu32 " bytes\n",
           free_heap_after);

    printf("Minimum heap before: %" PRIu32 " bytes\n",
           min_heap_before);

    printf("Minimum heap after:  %" PRIu32 " bytes\n",
           min_heap_after);

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
