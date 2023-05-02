/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/memory/generic_buffer.h"
#include "core/session/session.h"

void (*buffer_callback_fun)(const rocprofiler_record_header_t* begin,
                            const rocprofiler_record_header_t* end, rocprofiler_session_id_t session_id,
                            rocprofiler_buffer_id_t buffer_id);

/**
 * @brief This class creates a single timestamp session
 *
 */

class TimeStampSession : public ::testing::Test {
 protected:
  rocprofiler_session_id_t session_id{1234};
  std::unique_ptr<rocmtools::Session> session_ptr =
      std::make_unique<rocmtools::Session>(ROCPROFILER_NONE_REPLAY_MODE, session_id);

  void SetUp() {
    rocprofiler_filter_id_t filter_id =
        session_ptr->CreateFilter(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION,
                                  rocprofiler_filter_data_t{}, 0, rocprofiler_filter_property_t{});
    rocprofiler_buffer_id_t buffer_id = session_ptr->CreateBuffer(buffer_callback_fun, 0x9999);
    session_ptr->GetFilter(filter_id)->SetBufferId(buffer_id);
    session_ptr->Start();
  }
  void TearDown() { session_ptr->Terminate(); }
};

TEST_F(TimeStampSession, NewlyActivatedSessionIsActive) {
  // check if session is inactive
  EXPECT_TRUE(session_ptr->IsActive());
}

TEST_F(TimeStampSession, DeactivatingNewlyCreatedSessionPasses) {
  // check if session is inactive
  EXPECT_TRUE(session_ptr->IsActive());

  // Activate session
  session_ptr->Terminate();

  // check if session is active
  EXPECT_FALSE(session_ptr->IsActive());
}

TEST_F(TimeStampSession, DeactivatingAnActivatedSessionPasses) {
  // activate the session
  session_ptr->Start();

  // check if session is active
  EXPECT_TRUE(session_ptr->IsActive());

  // deactivate the session
  session_ptr->Terminate();

  // check if session is inactive
  EXPECT_FALSE(session_ptr->IsActive());
}

TEST_F(TimeStampSession, ForANewlyCreatedSessionValidSessionIdIsReturned) {
  // get session id
  rocprofiler_session_id_t session_id = session_ptr->GetId();

  // check for the valid id
  EXPECT_EQ(1234, session_id.handle);
}

/**
 * @brief This class creates multiple time stamp sessions
 *
 */
class TestingMultipleSessions : public ::testing::Test {
 protected:
  std::vector<std::unique_ptr<rocmtools::Session>> session_list;
  uint64_t number_of_sessions = 5;
  void SetUp() {
    for (uint64_t id = 0; id < number_of_sessions; id++) {
      std::unique_ptr<rocmtools::Session> timestamp_session = std::make_unique<rocmtools::Session>(
          ROCPROFILER_NONE_REPLAY_MODE, rocprofiler_session_id_t{id});

      rocprofiler_filter_id_t filter_id = timestamp_session->CreateFilter(
          ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION, rocprofiler_filter_data_t{}, 0,
          rocprofiler_filter_property_t{});
      rocprofiler_buffer_id_t buffer_id =
          timestamp_session->CreateBuffer(buffer_callback_fun, 0x9999);
      timestamp_session->GetFilter(filter_id)->SetBufferId(buffer_id);
      timestamp_session->Start();
      session_list.push_back(std::move(timestamp_session));
    }
  }
  void TearDown() {
    for (uint64_t id = 0; id < number_of_sessions; id++) {
      session_list[id]->Terminate();
    }
  }
};

TEST_F(TestingMultipleSessions, AllSessionsAreCreatedSuccessfully) {
  // check if sessions are inactive
  for (uint64_t id = 0; id < number_of_sessions; id++) {
    EXPECT_TRUE(session_list[id]->IsActive());
  }
}

TEST_F(TestingMultipleSessions, AllSessionsAreActivatedSuccessfully) {
  // Activate all sessions
  for (uint64_t id = 0; id < number_of_sessions; id++) {
    session_list[id]->Start();
  }
  // Check if sessions are activated
  for (uint64_t id = 0; id < number_of_sessions; id++) {
    EXPECT_TRUE(session_list[id]->IsActive());
  }
}

TEST_F(TestingMultipleSessions, DeactivatingAnActivatedSessionPasses) {
  // Activate all sessions
  for (uint64_t id = 0; id < number_of_sessions; id++) {
    session_list[id]->Start();
  }

  // Check if sessions are activated
  for (uint64_t id = 0; id < number_of_sessions; id++) {
    EXPECT_TRUE(session_list[id]->IsActive());
  }

  // deactivate the sessions
  for (uint64_t id = 0; id < number_of_sessions; id++) {
    session_list[id]->Terminate();
  }

  // check if all sessions are deactivated
  for (uint64_t id = 0; id < number_of_sessions; id++) {
    EXPECT_FALSE(session_list[id]->IsActive());
  }
}

// Createing sessions with 2 different profiling mode
TEST(WhenCreatingTwoSessionsWithDiffProfilingMode, BothSessionsAreCreated) {
  std::map<uint64_t, std::unique_ptr<rocmtools::Session>> sessions;

  sessions = std::map<uint64_t, std::unique_ptr<rocmtools::Session>>();

  {
    // create a counter collection session
    rocprofiler_session_id_t session_id{1};
    std::vector<const char*> counters;
    counters.emplace_back("SQ_WAVES");
    counters.emplace_back("GRBM_COUNT");
    sessions.emplace(session_id.handle,
                     std::make_unique<rocmtools::Session>(ROCPROFILER_NONE_REPLAY_MODE, session_id));

    rocprofiler_filter_id_t filter_id =
        sessions.at(session_id.handle)
            ->CreateFilter(ROCPROFILER_COUNTERS_COLLECTION,
                           rocprofiler_filter_data_t{.counters_names = &counters[0]}, counters.size(),
                           rocprofiler_filter_property_t{});
    rocprofiler_buffer_id_t buffer_id =
        sessions.at(session_id.handle)->CreateBuffer(buffer_callback_fun, 0x9999);
    sessions.at(session_id.handle)->GetFilter(filter_id)->SetBufferId(buffer_id);
  }
  {
    // create a timestamp collection session
    rocprofiler_session_id_t session_id{2};
    sessions.emplace(session_id.handle,
                     std::make_unique<rocmtools::Session>(ROCPROFILER_NONE_REPLAY_MODE, session_id));
    rocprofiler_filter_id_t filter_id =
        sessions.at(session_id.handle)
            ->CreateFilter(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION, rocprofiler_filter_data_t{}, 0,
                           rocprofiler_filter_property_t{});
    rocprofiler_buffer_id_t buffer_id =
        sessions.at(session_id.handle)->CreateBuffer(buffer_callback_fun, 0x9999);
    sessions.at(session_id.handle)->GetFilter(filter_id)->SetBufferId(buffer_id);
  }

  // check for correct profiling mode
  EXPECT_TRUE(sessions.at(1)->FindFilterWithKind(ROCPROFILER_COUNTERS_COLLECTION));
  EXPECT_TRUE(sessions.at(2)->FindFilterWithKind(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION));

  sessions.clear();
}