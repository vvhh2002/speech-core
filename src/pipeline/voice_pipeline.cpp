#include "speech_core/pipeline/voice_pipeline.h"
#include "speech_core/audio/pcm_codec.h"
#include "speech_core/audio/resampler.h"

#include <chrono>
#include <stdexcept>

namespace speech_core {

VoicePipeline::VoicePipeline(
    STTInterface& stt,
    TTSInterface& tts,
    LLMInterface* llm,
    VADInterface& vad,
    AgentConfig config,
    EventCallback on_event,
    EnhancerInterface* enhancer)
    : stt_(stt),
      tts_(tts),
      llm_(llm),
      enhancer_(enhancer),
      config_(config),
      on_event_(std::move(on_event)),
      turn_detector_(vad, config,
                     [this](const TurnEvent& e) { on_turn_event(e); }),
      context_(/* system_prompt */ "",
               config.max_history_messages > 0 ? config.max_history_messages : 0,
               config.max_history_tokens > 0 ? config.max_history_tokens : 0,
               config.mask_tool_results) {}

VoicePipeline::~VoicePipeline() {
    stop();
}

void VoicePipeline::start() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_.store(true);
        state_.store(State::Idle);
        is_synthesizing_.store(false);
        if (echo_canceller_) echo_canceller_->reset();
    }
    worker_thread_ = std::thread(&VoicePipeline::worker_loop, this);
}

void VoicePipeline::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_.store(false);
        tts_.cancel();
        if (llm_) llm_->cancel();
        speech_queue_.cancel_all();
        state_.store(State::Idle);
        is_synthesizing_.store(false);
    }
    worker_cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void VoicePipeline::resume_listening() {
    if (!running_.load()) return;
    auto s = state_.load();
    if (s == State::Speaking) {
        std::lock_guard<std::mutex> lock(mutex_);
        turn_detector_.set_agent_speaking(false);
        turn_detector_.reset();
        if (config_.post_playback_guard > 0) {
            turn_detector_.set_post_playback_guard(config_.post_playback_guard);
        }
        state_.store(State::Idle);
    }
}

void VoicePipeline::push_audio(const float* samples, size_t count) {
    if (!running_.load()) return;
    std::lock_guard<std::mutex> lock(mutex_);

    const float* audio = samples;

    if (echo_canceller_ && count > 0) {
        aec_buf_.resize(count);
        echo_canceller_->cancel_echo(audio, count, aec_buf_.data());
        audio = aec_buf_.data();
    }

    if (enhancer_ && count > 0) {
        enhance_buf_.resize(count);
        enhancer_->enhance(audio, count, enhancer_->input_sample_rate(),
                           enhance_buf_.data());
        audio = enhance_buf_.data();
    }

    turn_detector_.push_audio(audio, count);
}

void VoicePipeline::worker_loop() {
    if (config_.warmup_stt) {
        std::vector<float> silence(stt_.input_sample_rate() / 2, 0.0f);
        stt_.transcribe(silence.data(), silence.size(), stt_.input_sample_rate());
    }

    bool streaming_active = false;
    size_t stream_offset = 0;

    while (running_.load()) {
        PendingUtterance utterance;
        bool have_utterance = false;
        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            bool use_timed_wait = config_.emit_partial_transcriptions
                                  && stt_.supports_streaming()
                                  && !streaming_active;
            bool poll_stream = streaming_active;

            if (use_timed_wait || poll_stream) {
                auto timeout = std::chrono::milliseconds(
                    static_cast<int>(config_.partial_transcription_interval * 1000));
                worker_cv_.wait_for(lock, timeout, [this] {
                    return !pending_utterances_.empty() || !running_.load();
                });
            } else {
                worker_cv_.wait(lock, [this] {
                    return !pending_utterances_.empty() || !running_.load();
                });
            }
            if (!running_.load()) {
                if (streaming_active) {
                    stt_.cancel_stream();
                    streaming_active = false;
                }
                return;
            }

            if (!pending_utterances_.empty()) {
                worker_busy_.store(true);
                utterance = std::move(pending_utterances_.front());
                pending_utterances_.erase(pending_utterances_.begin());
                have_utterance = true;
            }
        }

        if (!have_utterance && streaming_active) {
            std::vector<float> snapshot;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (turn_detector_.in_speech()) {
                    snapshot = turn_detector_.utterance_snapshot();
                }
            }
            if (snapshot.size() > stream_offset) {
                try {
                    auto partial = stt_.push_chunk(
                        snapshot.data() + stream_offset,
                        snapshot.size() - stream_offset);
                    stream_offset = snapshot.size();
                    if (!partial.text.empty()) {
                        PipelineEvent event;
                        event.type = EventType::PartialTranscription;
                        event.text = partial.text;
                        event.confidence = partial.confidence;
                        on_event_(partial.text.empty() ? event : event); // Avoid unused warning
                        on_event_(event);
                    }
                } catch (...) {}
            }
            continue;
        }

        if (!have_utterance && !streaming_active
            && config_.emit_partial_transcriptions
            && stt_.supports_streaming()) {
            bool speech_active;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                speech_active = turn_detector_.in_speech();
            }
            if (speech_active) {
                stt_.begin_stream(stt_.input_sample_rate());
                streaming_active = true;
                stream_offset = 0;
            }
            continue;
        }

        if (!have_utterance) continue;

        {
            PipelineEvent ended;
            ended.type = EventType::SpeechEnded;
            ended.start_time = utterance.time;
            on_event_(ended);
        }

        try {
            auto stt_start = std::chrono::steady_clock::now();
            TranscriptionResult result;

            if (streaming_active) {
                if (utterance.audio.size() > stream_offset) {
                    stt_.push_chunk(
                        utterance.audio.data() + stream_offset,
                        utterance.audio.size() - stream_offset);
                }
                result = stt_.end_stream();
                streaming_active = false;
                stream_offset = 0;
            } else {
                result = stt_.transcribe(
                    utterance.audio.data(), utterance.audio.size(),
                    stt_.input_sample_rate());
            }

            float stt_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - stt_start).count();

            bool invalidated = eager_invalidated_.exchange(false);

            if (!invalidated) {
                PipelineEvent transcript_event;
                transcript_event.type = EventType::TranscriptionCompleted;
                transcript_event.text = result.text;
                transcript_event.start_time = utterance.time;
                transcript_event.stt_duration_ms = stt_ms;
                on_event_(transcript_event);
            }

            bool low_confidence = config_.min_transcription_confidence > 0 &&
                                  result.confidence < config_.min_transcription_confidence;

            if (!invalidated && !result.text.empty() && !low_confidence) {
                process_utterance(result.text, result.language, stt_ms);
            } else if (invalidated) {
                // Ignore
            } else {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    turn_detector_.set_agent_speaking(false);
                    turn_detector_.reset();
                }
                state_.store(State::Idle);
            }
        } catch (const std::exception& ex) {
            emit_error(std::string("STT failed: ") + ex.what());
            state_.store(State::Idle);
        }

        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            if (pending_utterances_.empty()) {
                worker_busy_.store(false);
                worker_idle_cv_.notify_all();
            }
        }
    }
}

void VoicePipeline::wait_idle() {
    std::unique_lock<std::mutex> lock(worker_mutex_);
    worker_idle_cv_.wait(lock, [this] {
        return (pending_utterances_.empty() && !worker_busy_.load()) || !running_.load();
    });
}

void VoicePipeline::push_text(const std::string& text) {
    if (!running_.load()) return;
    process_utterance(text);
}

void VoicePipeline::on_turn_event(const TurnEvent& event) {
    switch (event.type) {
    case TurnEvent::UserSpeechStarted: {
        if (event.eager_resumed) {
            eager_invalidated_.store(true);
            turn_detector_.set_agent_speaking(false);
        }
        state_.store(State::Listening);
        PipelineEvent e;
        e.type = EventType::SpeechStarted;
        e.start_time = event.time;
        on_event_(e);
        break;
    }

    case TurnEvent::UserSpeechEnded: {
        state_.store(State::Transcribing);
        {
            std::lock_guard<std::mutex> wlock(worker_mutex_);
            pending_utterances_.push_back({event.audio, event.time, event.eager});
        }
        worker_cv_.notify_one();
        break;
    }

    case TurnEvent::Interruption: {
        tts_.cancel();
        if (llm_) llm_->cancel();
        speech_queue_.cancel_all();
        turn_detector_.set_agent_speaking(false);
        is_synthesizing_.store(false);
        state_.store(State::Listening);

        PipelineEvent interrupted;
        interrupted.type = EventType::ResponseInterrupted;
        interrupted.start_time = event.time;
        on_event_(interrupted);
        break;
    }

    case TurnEvent::InterruptionRecovered:
        break;
    }
}

void VoicePipeline::process_utterance(const std::string& transcript,
                                      const std::string& language,
                                      float stt_duration_ms) {
    context_.add_user_message(transcript);

    std::string response_text;
    float llm_ms = 0.0f;

    switch (config_.mode) {
    case AgentConfig::Mode::Echo:
        speak(transcript, language, stt_duration_ms, 0.0f);
        break;

    case AgentConfig::Mode::TranscribeOnly:
        state_.store(State::Idle);
        return;

    case AgentConfig::Mode::Pipeline:
        if (!llm_) {
            speak(transcript, language, stt_duration_ms, 0.0f);
        } else {
            state_.store(State::Thinking);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                turn_detector_.set_agent_speaking(true);
            }

            try {
                auto llm_start = std::chrono::steady_clock::now();
                response_text = call_llm_with_tools();
                llm_ms = std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - llm_start).count();
            } catch (const std::exception& ex) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    turn_detector_.set_agent_speaking(false);
                }
                emit_error(std::string("LLM failed: ") + ex.what());
                state_.store(State::Idle);
                return;
            }

            if (state_.load() != State::Thinking && state_.load() != State::Speaking) {
                return;
            }
        }
        break;
    }

    if (!response_text.empty()) {
        context_.add_assistant_message(response_text);
        if (state_.load() == State::Thinking) {
            state_.store(State::Speaking);
        }
    } else if (config_.mode == AgentConfig::Mode::Pipeline) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            turn_detector_.set_agent_speaking(false);
        }
        state_.store(State::Idle);
    }
}

std::string VoicePipeline::call_llm_with_tools() {
    if (tool_registry_.size() > 0) {
        llm_->set_tools(tool_registry_.tools());
    }

    std::string accumulated;
    auto response = llm_->chat(context_.messages(),
        [this, &accumulated](const std::string& sentence, bool is_final) {
            if (is_final) return;
            if (!sentence.empty()) {
                accumulated += sentence;
                speak(sentence, "", 0.0f, 0.0f);
            }
        });

    return response.text.empty() ? accumulated : response.text;
}

void VoicePipeline::speak(const std::string& text, const std::string& language,
                          float stt_duration_ms, float llm_duration_ms) {
    if (state_.load() == State::Listening || !running_.load()) {
        return;
    }

    state_.store(State::Speaking);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        turn_detector_.set_agent_speaking(true);
    }

    speech_queue_.enqueue(text);

    // 异步触发队列处理，注意必须按值捕获以确保生命周期安全
    std::thread([this, language_copy = language, stt_duration_ms, llm_duration_ms]() {
        this->process_speech_queue(language_copy, stt_duration_ms, llm_duration_ms);
    }).detach();
}

void VoicePipeline::process_speech_queue(const std::string& language,
                                         float stt_duration_ms, float llm_duration_ms) {
    // 使用 exchange 确保只有一个线程进入合成流程
    if (is_synthesizing_.exchange(true)) {
        return;
    }

    while (running_.load()) {
        auto* item = speech_queue_.next();
        if (!item) {
            is_synthesizing_.store(false);
            // 再次检查，防止在标记 false 的瞬间有新项入队
            if (speech_queue_.size() > 0) {
                if (!is_synthesizing_.exchange(true)) continue;
            }
            break;
        }

        uint64_t speech_id = item->id;
        std::string text = item->text;
        const auto& tts_language = !language.empty() ? language : config_.language;

        try {
            size_t total_samples = 0;
            size_t max_samples = config_.max_response_duration > 0
                ? static_cast<size_t>(config_.max_response_duration * tts_.output_sample_rate())
                : 0;

            auto tts_start = std::chrono::steady_clock::now();
            
            // 使用同步等待模拟异步合成过程，确保顺序性
            std::promise<void> synthesis_promise;
            auto synthesis_future = synthesis_promise.get_future();

            tts_.synthesize(text, tts_language,
                [this, speech_id, &total_samples, max_samples,
                 tts_start, stt_duration_ms, llm_duration_ms, &synthesis_promise](
                    const float* samples, size_t length, bool is_final) {
                    
                    if (state_.load() == State::Listening) {
                        synthesis_promise.set_value(); // 即使打断也要通知结束
                        return;
                    }

                    size_t emit_length = length;
                    bool force_final = false;
                    if (max_samples > 0 && total_samples + length > max_samples) {
                        emit_length = max_samples - total_samples;
                        force_final = true;
                    }
                    total_samples += emit_length;

                    if (emit_length > 0) {
                        if (echo_canceller_) {
                            int tts_rate = tts_.output_sample_rate();
                            int aec_rate = echo_canceller_->input_sample_rate();

                            if (tts_rate != aec_rate) {
                                auto resampled = Resampler::resample(
                                    samples, emit_length, tts_rate, aec_rate);
                                echo_canceller_->feed_reference(
                                    resampled.data(), resampled.size());
                            } else {
                                echo_canceller_->feed_reference(samples, emit_length);
                            }
                        }

                        auto pcm = PCMCodec::float_to_pcm16(samples, emit_length);
                        PipelineEvent audio_event;
                        audio_event.type = EventType::ResponseAudioDelta;
                        audio_event.audio_data = std::move(pcm);
                        on_event_(audio_event);
                    }

                    if (is_final || force_final) {
                        speech_queue_.mark_done(speech_id);
                        
                        float tts_ms = std::chrono::duration<float, std::milli>(
                            std::chrono::steady_clock::now() - tts_start).count();

                        PipelineEvent done;
                        done.type = EventType::ResponseDone;
                        done.stt_duration_ms = stt_duration_ms;
                        done.llm_duration_ms = llm_duration_ms;
                        done.tts_duration_ms = tts_ms;
                        on_event_(done);

                        if (force_final && !is_final) {
                            tts_.cancel();
                        }
                        
                        try {
                            synthesis_promise.set_value();
                        } catch (...) {}
                    }
                });
            
            // 等待当前片段合成播放结束
            synthesis_future.wait();
            
        } catch (const std::exception& ex) {
            speech_queue_.mark_done(speech_id);
            emit_error(std::string("TTS failed: ") + ex.what());
        }
        
        if (state_.load() == State::Listening) break;
    }
    
    is_synthesizing_.store(false);
}

void VoicePipeline::emit_error(const std::string& message) {
    PipelineEvent error;
    error.type = EventType::Error;
    error.text = message;
    on_event_(error);
}

}  // namespace speech_core
