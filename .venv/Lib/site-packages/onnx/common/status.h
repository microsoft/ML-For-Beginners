// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <utility>

namespace ONNX_NAMESPACE {
namespace Common {

enum StatusCategory {
  NONE = 0,
  CHECKER = 1,
  OPTIMIZER = 2,
};

enum StatusCode {
  OK = 0,
  FAIL = 1,
  INVALID_ARGUMENT = 2,
  INVALID_PROTOBUF = 3,
};

class Status {
 public:
  Status() noexcept {}

  Status(StatusCategory category, int code, const std::string& msg);

  Status(StatusCategory category, int code);

  Status(const Status& other) {
    *this = other;
  }

  void operator=(const Status& other) {
    if (&other != this) {
      if (nullptr == other.state_) {
        state_.reset();
      } else if (state_ != other.state_) {
        state_.reset(new State(*other.state_));
      }
    }
  }

  Status(Status&&) = default;
  Status& operator=(Status&&) = default;
  ~Status() = default;

  bool IsOK() const noexcept;

  int Code() const noexcept;

  StatusCategory Category() const noexcept;

  const std::string& ErrorMessage() const;

  std::string ToString() const;

  bool operator==(const Status& other) const {
    return (this->state_ == other.state_) || (ToString() == other.ToString());
  }

  bool operator!=(const Status& other) const {
    return !(*this == other);
  }

  static const Status& OK() noexcept;

 private:
  struct State {
    State(StatusCategory cat_, int code_, std::string msg_) : category(cat_), code(code_), msg(std::move(msg_)) {}

    StatusCategory category = StatusCategory::NONE;
    int code = 0;
    std::string msg;
  };

  static const std::string& EmptyString();

  // state_ == nullptr when if status code is OK.
  std::unique_ptr<State> state_;
};

inline std::ostream& operator<<(std::ostream& out, const Status& status) {
  return out << status.ToString();
}

} // namespace Common
} // namespace ONNX_NAMESPACE
