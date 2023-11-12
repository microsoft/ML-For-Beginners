// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <stdint.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "onnx/common/array_ref.h"
#include "onnx/common/assertions.h"
#include "onnx/common/common.h"
#include "onnx/common/graph_node_list.h"
#include "onnx/common/interned_strings.h"
#include "onnx/common/tensor.h"
#include "onnx/string_utils.h"

#define ONNX_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;           \
  TypeName& operator=(const TypeName&) = delete

namespace ONNX_NAMESPACE {

// Graph represents one "function" of computation.
// It uses a simple ownership model where the graph owns all the nodes inside it.
// All references inside the graph are raw pointers.
// Destroying the Graph will invalidate any pointers to nodes in the graph.
struct Graph;

// Node is the base class of the IR graph. It represents one computation
// and dependencies on a list of Values. The "prim-ops", so to speak.
struct Node;

// A Value represents an input or output to node that is either a
// Tensor or an opaque Handle object, as determined by type().
struct Value;

class ResourceGuard final {
  std::function<void()> destructor_;
  bool released_;

 public:
  ONNX_DISALLOW_COPY_AND_ASSIGN(ResourceGuard);
  explicit ResourceGuard(std::function<void()> destructor) : destructor_(std::move(destructor)), released_(false) {}
  ResourceGuard(ResourceGuard&& other) = default;
  ResourceGuard& operator=(ResourceGuard&& other) = default;

  ~ResourceGuard() {
    if (!released_)
      destructor_();
  }

  void release() {
    released_ = true;
  }
};

struct Dimension final {
  Dimension() : is_unknown(true), is_int(false), dim(-1) {}
  Dimension(std::string param) : is_unknown(false), is_int(false), dim(-1), param(std::move(param)) {} // NOLINT
  Dimension(int64_t dim) : is_unknown(false), is_int(true), dim(dim) {} // NOLINT

  bool is_unknown;
  bool is_int;
  int64_t dim;
  std::string param;
};

enum class AttributeKind : uint8_t {
  // float, float list, int, int list, string, string list,
  // tensor, tensor list, subgraph, subgraph list. type proto, type proto list
  f,
  fs,
  i,
  is,
  s,
  ss,
  t,
  ts,
  g,
  gs,
  tp,
  tps
};

static inline const char* toString(AttributeKind kind) {
  static constexpr const char* names[] = {"f", "fs", "i", "is", "s", "ss", "t", "ts", "g", "gs", "tp", "tps"};
  ONNX_ASSERT(size_t(kind) < sizeof(names) / sizeof(const char*));
  return names[int(kind)];
}

struct AttributeValue {
  explicit AttributeValue(Symbol name) : name(name) {}
  using Ptr = std::unique_ptr<AttributeValue>;
  Symbol name;
  virtual AttributeKind kind() const = 0;
  virtual Ptr clone() const = 0;
  virtual ~AttributeValue() = default;
};

template <typename T, AttributeKind Kind>
struct ScalarAttributeValue final : public AttributeValue {
  using ConstructorType = const T&;
  using ValueType = T;
  ScalarAttributeValue(Symbol name, ConstructorType value_) : AttributeValue(name), value_(value_) {}
  ValueType& value() {
    return value_;
  }
  virtual Ptr clone() const override {
    return Ptr(new ScalarAttributeValue(name, value_));
  }
  virtual AttributeKind kind() const override {
    return Kind;
  }

 private:
  ValueType value_;
};

template <typename T, AttributeKind Kind>
struct VectorAttributeValue final : public AttributeValue {
  using ConstructorType = const std::vector<T>&&;
  using ValueType = std::vector<T>;
  VectorAttributeValue(Symbol name, ConstructorType value_) : AttributeValue(name), value_(std::move(value_)) {}
  ValueType& value() {
    return value_;
  }
  virtual AttributeKind kind() const override {
    return Kind;
  }
  virtual std::unique_ptr<AttributeValue> clone() const override {
    auto copy = value_;
    return Ptr(new VectorAttributeValue(name, std::move(copy)));
  }

 private:
  ValueType value_;
};

using FloatAttr = ScalarAttributeValue<double, AttributeKind::f>;
using FloatsAttr = VectorAttributeValue<double, AttributeKind::fs>;
using IntAttr = ScalarAttributeValue<int64_t, AttributeKind::i>;
using IntsAttr = VectorAttributeValue<int64_t, AttributeKind::is>;
using StringAttr = ScalarAttributeValue<std::string, AttributeKind::s>;
using StringsAttr = VectorAttributeValue<std::string, AttributeKind::ss>;
using TensorAttr = ScalarAttributeValue<Tensor, AttributeKind::t>;
using TensorsAttr = VectorAttributeValue<Tensor, AttributeKind::ts>;
using GraphAttr = ScalarAttributeValue<std::shared_ptr<Graph>, AttributeKind::g>;
using GraphsAttr = VectorAttributeValue<std::shared_ptr<Graph>, AttributeKind::gs>;
using TypeProtoAttr = ScalarAttributeValue<TypeProto, AttributeKind::tp>;
using TypeProtosAttr = VectorAttributeValue<TypeProto, AttributeKind::tps>;

// CRTP so that Node which inherits Attributes can be return for
// method chaining e.g:
// Node * n = g->create(kSelect)->set_i(kOffset,3)->set_f(kValue,3.5);
// we return Derived* pointers because Nodes are normally held as pointers.
template <typename Derived>
struct Attributes {
  Attributes() {}
  void copyAttributes(const Attributes& rhs) {
    values_.clear();
    values_.reserve(rhs.values_.size());
    for (auto& i : rhs.values_) {
      values_.push_back(i->clone());
    }
  }
  bool hasAttribute(Symbol name) const {
    return find(name, false) != values_.end();
  }
  AttributeKind kindOf(Symbol name) const {
    return (*find(name, true))->kind();
  }
  Derived* removeAttribute(Symbol name) {
    values_.erase(find(name, true));
    return This();
  }
  bool hasAttributes() const {
    return !values_.empty();
  }
  // The names are returned in order, since name actually is the index.
  std::vector<Symbol> attributeNames() const {
    std::vector<Symbol> names;
    names.reserve(values_.size());
    for (auto& a : values_)
      names.push_back(a->name);
    return names;
  }

#define CREATE_ACCESSOR(Kind, method)                                           \
  Derived* method##_(Symbol name, Kind##Attr::ConstructorType v) {              \
    return set<Kind##Attr>(name, std::forward<Kind##Attr::ConstructorType>(v)); \
  }                                                                             \
  const Kind##Attr::ValueType& method(Symbol name) const {                      \
    return get<Kind##Attr>(name);                                               \
  }
  CREATE_ACCESSOR(Float, f)
  CREATE_ACCESSOR(Floats, fs)
  CREATE_ACCESSOR(String, s)
  CREATE_ACCESSOR(Strings, ss)
  CREATE_ACCESSOR(Int, i)
  CREATE_ACCESSOR(Ints, is)
  CREATE_ACCESSOR(Tensor, t)
  CREATE_ACCESSOR(Tensors, ts)
  CREATE_ACCESSOR(Graph, g)
  CREATE_ACCESSOR(Graphs, gs)
  CREATE_ACCESSOR(TypeProto, tp)
  CREATE_ACCESSOR(TypeProtos, tps)

#undef CREATE_ACCESSOR

 private:
  Derived* This() {
    return static_cast<Derived*>(this);
  }
  template <typename T>
  Derived* set(Symbol name, typename T::ConstructorType v) {
    auto it = find(name, false);
    auto nv = AVPtr(new T(name, std::forward<typename T::ConstructorType>(v)));
    if (it == values_.end()) {
      values_.push_back(std::move(nv));
    } else {
      *it = std::move(nv);
    }
    return This();
  }
  template <typename T>
  typename T::ValueType& get(Symbol name) const {
    auto it = find(name, true);
    T* child = static_cast<T*>(it->get());
    return child->value();
  }
  using AVPtr = AttributeValue::Ptr;
  // NB: For determinism, we use a vector rather than a hash map.  This does
  // mean that lookups are O(n), so you shouldn't use Attributes to store
  // a big pile of messages.
  std::vector<AVPtr> values_;
  using iterator = std::vector<AVPtr>::iterator;
  iterator find(Symbol name, bool required) {
    auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) { return v->name == name; });
    ONNX_ASSERT(!required || it != values_.end());
    return it;
  }
  using const_iterator = std::vector<AVPtr>::const_iterator;
  const_iterator find(Symbol name, bool required) const {
    auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) { return v->name == name; });
    ONNX_ASSERTM(
        !required || it != values_.end(),
        "%s:%u: %s: required undefined attribute '%s'",
        __FILE__,
        __LINE__,
        __func__,
        name.toString());
    return it;
  }
};

// Each use is represented by this type, see Node::uses()
// 'user' is the consumer of the value, offset is the index into
// 'user's input this where the produces will be found.
struct Use final {
  Use(Node* user, size_t offset) : user(user), offset(offset) {}
  Node* user;
  size_t offset;
};

static inline bool operator==(const Use& a, const Use& b) {
  return a.user == b.user && a.offset == b.offset;
}

// the list types are intentionally simple, but we type-def
// them here so if we need to change them, refactoring will be easier
using node_list = std::vector<Node*>;
using value_list = std::vector<Value*>;
using use_list = std::vector<Use>;
using NodeKind = Symbol;

struct Value final {
  ONNX_DISALLOW_COPY_AND_ASSIGN(Value);
  Value(Node* node_, size_t offset_);
  Value(Value&&) = default;
  Value& operator=(Value&&) = default;
  ~Value() = default;

 private:
  friend struct Node;
  friend struct Graph;
  Node* node_;
  size_t offset_;
  size_t unique_ = 0; // unique id
  size_t stage_ = 0; // 0-forward, 1-backward, 2-double-backward,...
  use_list uses_in_current_graph_;
  bool has_unique_name_;
  std::string unique_name_;
  int32_t elem_type_;
  bool has_sizes_;
  std::vector<Dimension> sizes_;

 public:
  Value* setElemType(int32_t elem_type) {
    elem_type_ = elem_type;
    return this;
  }
  int32_t elemType() const {
    return elem_type_;
  }
  bool has_sizes() const {
    return has_sizes_;
  }
  Value* setSizes(std::vector<Dimension> sizes) {
    has_sizes_ = true;
    sizes_ = std::move(sizes);
    return this;
  }
  Value* wipeSizes() {
    has_sizes_ = false;
    sizes_ = std::vector<Dimension>();
    return this;
  }
  const std::vector<Dimension>& sizes() const {
    return sizes_;
  }
  size_t unique() const {
    return unique_;
  }
  bool has_unique_name() const {
    return has_unique_name_;
  }
  std::string uniqueName() const {
    if (has_unique_name())
      return unique_name_;
    return ONNX_NAMESPACE::to_string(unique());
  }
  Value* setUniqueName(const std::string& name, bool rename_subgraph_captured_nodes = true);
  Value* setStage(size_t s) {
    stage_ = s;
    return this;
  }
  size_t stage() const {
    return stage_;
  }
  Node* node() {
    return node_;
  }
  size_t offset() const {
    return offset_;
  }
  const Node* node() const {
    return node_;
  }
  Graph* owningGraph();
  const Graph* owningGraph() const;
  // TODO: make this more const correct
  const use_list uses() const;

  // Replaces all uses of this node with 'newValue'.
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%3, %3)
  // Execute: %3.replaceAllUsesWith(%6)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%6)
  //          %5 = h(%6, %6)
  void replaceAllUsesWith(Value* newValue);

  Value* copyMetadata(Value* from) {
    setElemType(from->elemType());
    setSizes(from->sizes());
    if (from->has_unique_name()) {
      setUniqueName(from->uniqueName());
    }
    return this;
  }
};

struct Node : public Attributes<Node> {
  ONNX_DISALLOW_COPY_AND_ASSIGN(Node);
  friend struct Graph;
  friend struct Value;
  friend graph_node_list;
  friend const_graph_node_list;
  friend graph_node_list_iterator;
  friend const_graph_node_list_iterator;

 private:
  // each node but Return/Param
  // is associated with exactly one place in the node list...
  // of the graph_
  // this circular is a doubly-linked list, the Return node is used as the sentinel for the beginning and end of the
  // list such that the list never has null pointers next_in_graph[0] is next pointer next_in_graph[1] is prev pointer
  // using an array to allow the same iterator class for forward and reverse node lists
  // This list represents a topological sort

  Node* next_in_graph[2] = {nullptr, nullptr};
  Node*& next() {
    return next_in_graph[kNextDirection];
  }
  Node*& prev() {
    return next_in_graph[kPrevDirection];
  }
  Node* const& next() const {
    return next_in_graph[kNextDirection];
  }
  Node* const& prev() const {
    return next_in_graph[kPrevDirection];
  }

  const NodeKind kind_;
  std::vector<Value*> inputs_;
  std::vector<Value*> outputs_;
  Graph* graph_;
  size_t stage_;
  bool has_name_;
  std::string name_;
  bool has_domain_;
  std::string domain_;
  bool has_doc_string_;
  std::string doc_string_;

 protected:
  Node(Graph* graph_, NodeKind kind_); // defined after graph

 public:
  bool has_name() const {
    return has_name_;
  }
  const std::string& name() const {
    return name_;
  }
  void setName(std::string name) {
    has_name_ = true;
    name_ = std::move(name);
  }
  bool has_domain() const {
    return has_domain_;
  }
  const std::string& domain() const {
    return domain_;
  }
  void setDomain(std::string domain) {
    has_domain_ = true;
    domain_ = std::move(domain);
  }
  bool has_doc_string() const {
    return has_doc_string_;
  }
  const std::string& docString() const {
    return doc_string_;
  }
  void setDocString(std::string doc_string) {
    has_doc_string_ = true;
    doc_string_ = std::move(doc_string);
  }
  NodeKind kind() const {
    return kind_;
  }
  Graph* owningGraph() {
    return graph_;
  }
  const Graph* owningGraph() const {
    return graph_;
  }
  size_t stage() const {
    return stage_;
  }
  Node* setStage(size_t s) {
    stage_ = s;
    return this;
  }
  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  ArrayRef<Value*> inputs() {
    return inputs_;
  }
  ArrayRef<const Value*> inputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {inputs_.data(), inputs_.size()};
  }
  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  ArrayRef<Value*> outputs() {
    return outputs_;
  }
  ArrayRef<const Value*> outputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {outputs_.data(), outputs_.size()};
  }
  bool hasUses() const {
    for (auto o : outputs()) {
      if (!o->uses().empty())
        return true;
    }
    return false;
  }
  void replaceAllUsesWith(Node* n) {
    ONNX_ASSERT(outputs().size() == n->outputs().size());
    size_t nOutputs = outputs().size();
    for (size_t i = 0; i < nOutputs; i++) {
      outputs()[i]->replaceAllUsesWith(n->outputs()[i]);
    }
  }
  // lots of things like chunk have a single input or single output, so we have a
  // helper to make accessing it easier
  Value* input() {
    ONNX_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  Value* output() {
    ONNX_ASSERT(outputs_.size() == 1);
    return outputs_.at(0);
  }
  const Value* input() const {
    ONNX_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  Value* output() const {
    ONNX_ASSERT(outputs_.size() == 1);
    return outputs_.at(0);
  }
  // Access a particular input.  This is a checked index.
  Value* input(size_t i) {
    return inputs_.at(i);
  }
  const Value* input(size_t i) const {
    return inputs_.at(i);
  }

  // Graphs

  // Note [Topological invariant]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // We always maintain an up-to-date topological ordering of all nodes via
  // the next()/prev() links.  All transformations to graphs must preserve
  // this topological ordering: for example, it is only valid to 'addInput'
  // with an input which is topologically before the current node.
  //
  // Usually, it is obvious whether or not topological order is maintained;
  // for example, if you are adding nodes to the end of the topsort, it's
  // impossible for them to refer to inputs that are not in the topsort.
  // If it is not obvious, please comment accordingly.

  // Add 'node' as an input to 'this' at the end of existing
  // arguments.  Returns the added node for ease of chaining.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.addInput(%4)
  // Result:  %3 = f(%1, %2, %4)
  Value* addInput(Value* node) {
    ONNX_ASSERT(graph_ == node->owningGraph());
    node->uses_in_current_graph_.emplace_back(this, inputs_.size());
    inputs_.push_back(node);
    return node;
  }

  // Replace the input of 'this' at position 'i' with
  // 'newValue', returning the old node.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.replaceInput(1, %4)
  // Result:  %3 = f(%1, %4)
  Value* replaceInput(size_t i, Value* newValue) {
    ONNX_ASSERT(newValue->owningGraph() == graph_);
    Value* old = dropInput(i);
    inputs_[i] = newValue;
    newValue->uses_in_current_graph_.emplace_back(this, i);
    return old;
  }

  // Replace all occurrences of 'from' in the inputs of this
  // node with 'to'. Corresponds to llvm's replaceUsesOfWith.
  //
  // Given:   %3 = f(%1, %2, %1)
  // Execute: %3.replaceInputWith(%1, %4)
  // Result:  %3 = f(%4, %2, %4)
  void replaceInputWith(Value* from, Value* to) {
    ONNX_ASSERT(from->owningGraph() == graph_);
    ONNX_ASSERT(to->owningGraph() == graph_);
    size_t i = 0;
    for (auto input : inputs()) {
      if (input == from)
        replaceInput(i, to);
      i++;
    }
  }

  Value* addOutput() {
    outputs_.push_back(new Value(this, outputs_.size()));
    return outputs_.back();
  }

  void eraseOutput(size_t i);

  // Insert unattached 'this' node after 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertBefore(%4)
  // Result:  %3 = f(%1, %2)
  //          %5 = h(%1)
  //          %4 = g(%3)
  Node* insertBefore(Node* n) {
    ONNX_ASSERT(n->inGraphList());
    insertAfter(n->prev());
    return this;
  }

  // Insert unattached 'this' node after 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given: %3 = f(%1, %2)
  //        %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertAfter(%4)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%1)
  Node* insertAfter(Node* n) {
    ONNX_ASSERT(!inGraphList() && n->inGraphList());
    Node* next = n->next();
    n->next() = this;
    this->prev() = n;
    this->next() = next;
    next->prev() = this;
    return this;
  }

  // Move 'this' (already in the graph) after 'n' in the topological order.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.moveAfter(%3)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  //
  void moveAfter(Node* n) {
    removeFromList();
    insertAfter(n);
  }

  // Move a node 'n' (already in the graph) before 'this' in the topological order.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %3.moveBefore(%2)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  void moveBefore(Node* n) {
    removeFromList();
    insertBefore(n);
  }

  // Remove the input at 'i' from this node.
  //
  // WARNING: This is O(n) in the number of inputs, so avoid repeatedly calling
  // removeInput.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeInput(1)
  // Result: %3 = f(%1)
  void removeInput(size_t i) {
    dropInput(i);
    // everything after this input shifts left,
    // so we need to update their use offsets to match
    for (size_t j = i + 1; j < inputs_.size(); j++) {
      auto it = findUseForInput(j);
      it->offset--;
    }
    inputs_.erase(inputs_.begin() + i);
  }

  // Remove all inputs from a node.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeAllInputs()
  // Result: %3 = f()
  void removeAllInputs() {
    for (size_t i = 0; i < inputs().size(); ++i)
      dropInput(i);
    inputs_.clear();
  }

  // Check whether this node is before node n in the graph.
  bool isBefore(Node* n);

  // iterators of the node list starting at this node
  // useful for resuming a search starting at this node
  graph_node_list_iterator iterator();
  graph_node_list_iterator reverseIterator();
  const_graph_node_list_iterator iterator() const;
  const_graph_node_list_iterator reverseIterator() const;

  // Remove 'this' from the instruction list and deallocate it.
  //
  // Invariant: no outputs of 'this' may have any uses.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.destroy()
  // Result: %3 = g(%1)
  void destroy();

  // Dynamically cast this node to the subclass indicated by the
  // template variable, returning nullptr if the cast is invalid..
  //
  // Example usage: if(auto s = n.cast<Select>()) { ... }
  //
  // TODO: Make this const correct
  template <typename T>
  T* cast() {
    if (T::Kind == kind())
      return static_cast<T*>(this);
    return nullptr;
  }
  template <typename T>
  T* expect() {
    ONNX_ASSERTM(T::Kind == kind(), "expected a %s but found a %s", T::Kind.toString(), kind().toString());
    return static_cast<T*>(this);
  }

  virtual ~Node() = default;

 private:
  // Lookup iterator in use list of _input i_ that corresponds to its use of _this_
  use_list::iterator findUseForInput(size_t i) {
    auto& input_uses = inputs_[i]->uses_in_current_graph_;
    // O(N) on the use list, but unless we get nodes with +100 uses
    // vector traversal still is probably faster than linked list
    auto use_it = std::find(input_uses.begin(), input_uses.end(), Use(this, i));
    ONNX_ASSERT(use_it != input_uses.end());
    return use_it;
  }

  // remove the use of input i, this sets input i to nullptr, but
  // is only used internally to Node before setting it to a new value
  // or erasing the entry from the list.
  Value* dropInput(size_t i) {
    ONNX_ASSERT(i < inputs_.size());
    auto input_node = inputs_[i];
    auto use_it = findUseForInput(i);
    input_node->uses_in_current_graph_.erase(use_it);
    inputs_[i] = nullptr;
    return input_node;
  }

  bool inGraphList() const {
    ONNX_ASSERT(next() != nullptr || prev() == nullptr);
    return next() != nullptr;
  }
  void removeFromList() {
    ONNX_ASSERT(inGraphList());
    Node* next = this->next();
    Node* prev = this->prev();
    prev->next() = next;
    next->prev() = prev;
    this->next() = nullptr;
    this->prev() = nullptr;
  }

 protected:
  // subclasses must override
  // this function is used by createClone to initialize a new version
  // of a node in another graph. It should allocate a new instance of the same
  // concrete type as 'this', but in graph 'g' which might be different
  // than graph_
  virtual Node* allocNewInstance(Graph* g) {
    return new Node(g, kind());
  }
  // create a copy of all properties of Node s into this.
  // subclasses should extend if they have additional information to copy.
  // 'this' will be allocated with s->allocNewInstance(g) so it should have
  // the same concrete type as 's'
  //
  // NB: This does NOT clone stages.  You're expected to set the stage correctly
  // if you are going to preserve it.
  virtual void cloneFrom(Node* s) {
    copyAttributes(*s);
  }
};

// A class with the same properties as OperatorSetIdProto, but without protobuf
// overhead, resulting in a simpler and more readable workflow.
class OpSetID final {
 private:
  std::string domain_;
  int64_t version_;

 public:
  explicit OpSetID(const OperatorSetIdProto& proto) : domain_(proto.domain()), version_(proto.version()) {}

  // Default Domain Constructor
  explicit OpSetID(const int64_t version) : domain_(""), version_(version) {}

  explicit OpSetID(const std::string& domain, int64_t version) : domain_(domain), version_(version) {}

  // target must be in the form "<domain>&<version>"
  std::string toString() const {
    return domain_ + "$" + ONNX_NAMESPACE::to_string(version_);
  }

  // target must be in the form "<domain>&<version>"
  static OpSetID fromString(const std::string& target) {
    ONNX_TRY {
      std::string new_domain = target.substr(0, target.find("$"));
      int new_version = ONNX_NAMESPACE::stoi(target.substr(target.find("$") + 1, target.length()).c_str());
      return OpSetID(new_domain, new_version);
    }
    ONNX_CATCH(const std::runtime_error& e) {
      ONNX_HANDLE_EXCEPTION([&]() { ONNX_ASSERTM(false, "Error in fromString: %s", e.what()); });
    }

    // The control will never reach here.
    // In the default build where exceptions are turned on in case of any error
    // the control will enter catch block where an exception will be thrown again.
    // In case of "no exception build" the code aborts at the site of first exception.
    // Adding this to appease the warning "control may reach end of non-void function"
    // as the mac build fails when ONNX_WERROR==ON
    return OpSetID("", 0);
  }

  const std::string& domain() const {
    return domain_;
  }

  int64_t version() const {
    return version_;
  }

  void incrementVersion(int64_t step) {
    version_ += step;
  }

  void setVersion(int64_t newVal) {
    version_ = newVal;
  }
};

struct Graph final {
  ONNX_DISALLOW_COPY_AND_ASSIGN(Graph);
  friend struct Node;
  friend struct Value;

 private:
  // only used to keep track of allocated nodes
  // actual representation of Graph is done with
  // inputs, outputs, nodes

  std::unordered_set<const Node*> all_nodes;
  std::unordered_set<const Value*> all_values;
  size_t next_unique_;

  size_t new_node_stage_;

  // holds outputs in a way that can be reflected
  // as a Use object
  // also used as the beginning/end of the circular node list to avoid
  // having corner cases where the list is empty.
  Node* const output_;
  Node* const input_;
  // Create an independent node list for those initializers do not exist in input
  Node* const initializer_node_;

  std::vector<Tensor> initializers_;
  std::vector<std::string> initializer_names_;

  bool has_name_;
  std::string name_;
  bool has_doc_string_;
  std::string doc_string_;

  std::vector<OpSetID> opset_versions_;

  bool isNameUnique(const std::string& name) const {
    if (std::find(initializer_names_.cbegin(), initializer_names_.cend(), name) != initializer_names_.cend()) {
      return false;
    }
    const auto f = [&name](const Value* v) { return v->uniqueName() == name; };
    for (const Node* node : all_nodes) {
      for (const auto& attr : node->attributeNames()) {
        if (node->kindOf(attr) == AttributeKind::g) {
          const auto& subgraph = node->g(attr);
          if (!subgraph->isNameUnique(name)) {
            return false;
          }
        } else if (node->kindOf(attr) == AttributeKind::gs) {
          for (const auto& subgraph : node->gs(attr)) {
            if (!subgraph->isNameUnique(name)) {
              return false;
            }
          }
        }
      }
      const auto found_in = std::find_if(node->inputs().begin(), node->inputs().end(), f);
      if (found_in != node->inputs().end()) {
        return false;
      }
      const auto found_out = std::find_if(node->outputs().begin(), node->outputs().end(), f);
      if (found_out != node->outputs().end()) {
        return false;
      }
    }
    return true;
  }

 public:
  Graph()
      : next_unique_(0),
        new_node_stage_(0),
        output_(initOutput(create(kReturn, 0))),
        input_(create(kParam, 0)),
        initializer_node_(create(kParam, 0)),
        has_name_(false),
        has_doc_string_(false) {}

  bool has_doc_string() const {
    return has_doc_string_;
  }
  const std::string& docString() {
    return doc_string_;
  }
  void setDocString(std::string doc_string) {
    has_doc_string_ = true;
    doc_string_ = std::move(doc_string);
  }

  void addInitializer(Tensor& initializer) {
    if (initializer.name().empty()) {
      initializer.setName(ONNX_NAMESPACE::to_string(getNextUnique()));
    }
    initializers_.push_back(initializer);
    initializer_names_.push_back(initializer.name());
  }

  // For IR >= 4, initializer is not required to exist in input
  // Add initializer into initializer node list and return its Value
  Value* addInitializerAndCreateValue(Tensor& initializer) {
    addInitializer(initializer);
    auto* init_value = initializer_node_->addOutput();
    std::vector<Dimension> dim_sizes{initializer.sizes().cbegin(), initializer.sizes().cend()};
    init_value->setUniqueName(initializer.name());
    init_value->setSizes(dim_sizes);
    init_value->setElemType(initializer.elem_type());
    return init_value;
  }

  void eraseInitializer(const std::string& name) {
    initializers_.erase(
        std::remove_if(
            initializers_.begin(),
            initializers_.end(),
            [&name](Tensor& initializer) { return initializer.name() == name; }),
        initializers_.end());
    initializer_names_.erase(
        std::remove(initializer_names_.begin(), initializer_names_.end(), name), initializer_names_.end());
    for (size_t i = 0; i < initializer_node_->outputs().size(); i++) {
      if (initializer_node_->outputs()[i]->uniqueName() == name) {
        initializer_node_->eraseOutput(i);
        break;
      }
    }
  }
  void clearInitializers() {
    initializers_.clear();
    initializer_names_.clear();
  }
  const std::vector<Tensor>& initializers() const {
    return initializers_;
  }
  const std::vector<std::string>& initializer_names() const {
    return initializer_names_;
  }
  std::vector<Tensor>::const_iterator getInitializer(const std::string& name) const {
    for (auto it = initializers_.cbegin(); it != initializers_.cend(); ++it) {
      if (name == it->name()) {
        return it;
      }
    }
    return initializers_.end();
  }
  bool is_constant_initializer(const Value* value) const {
    return value->node() == initializer_node_;
  }
  ArrayRef<Value*> inputs() {
    return input_->outputs();
  }
  ArrayRef<const Value*> inputs() const {
    const auto& inputs = input_->outputs();
    return {inputs.data(), inputs.size()};
  }
  ArrayRef<Value*> outputs() {
    return output_->inputs();
  }
  ArrayRef<const Value*> outputs() const {
    return static_cast<const Node*>(output_)->inputs();
  }
  graph_node_list nodes() {
    return graph_node_list(output_, kNextDirection);
  }
  const_graph_node_list nodes() const {
    return const_graph_node_list(output_, kNextDirection);
  }

  std::vector<OpSetID>& opset_versions_mutable() {
    return opset_versions_;
  }

  size_t getNextUnique() {
    std::string next_unique_name = ONNX_NAMESPACE::to_string(++next_unique_);
    while (!isNameUnique(next_unique_name)) {
      next_unique_name = ONNX_NAMESPACE::to_string(++next_unique_);
    }
    return next_unique_;
  }

  // These invocations of begin() on output of function are OK
  // because graph_node_list is non-owning, so it doesn't matter
  // if it immediately dies after the invocation.
  graph_node_list_iterator begin() {
    return nodes().begin();
  }
  const_graph_node_list_iterator begin() const {
    return nodes().begin();
  }
  graph_node_list_iterator end() {
    return nodes().end();
  }
  const_graph_node_list_iterator end() const {
    return nodes().end();
  }
  graph_node_list_iterator rbegin() {
    return nodes().rbegin();
  }
  const_graph_node_list_iterator rbegin() const {
    return nodes().rbegin();
  }
  graph_node_list_iterator rend() {
    return nodes().rend();
  }
  const_graph_node_list_iterator rend() const {
    return nodes().rend();
  }
  Node* return_node() {
    return output_;
  }
  const Node* return_node() const {
    return output_;
  }

  Value* addInput() {
    return input_->addOutput();
  }
  void eraseInput(size_t i) {
    input_->eraseOutput(i);
  }
  void advanceStage() {
    new_node_stage_++;
  }
  void setStage(size_t new_stage) {
    new_node_stage_ = new_stage;
  }
  size_t stage() const {
    return new_node_stage_;
  }
  ResourceGuard setStageTemporary(size_t s) {
    auto prev_stage = new_node_stage_;
    new_node_stage_ = s;
    return ResourceGuard([prev_stage, this]() { this->new_node_stage_ = prev_stage; });
  }

  size_t registerOutput(Value* n) {
    output_->addInput(n);
    return outputs().size() - 1;
  }

  Node* create(NodeKind kind, size_t num_outputs = 1) {
    // NB: Node constructor adds node to all_nodes
    auto n = new Node(this, kind);
    for (size_t i = 0; i < num_outputs; i++)
      n->addOutput();
    return n;
  }

  Node* create(NodeKind kind, ArrayRef<Value*> inputs, size_t num_outputs = 1) {
    auto n = create(kind, num_outputs);
    for (auto i : inputs)
      n->addInput(i);
    return n;
  }

  Node* appendNode(Node* n) {
    ONNX_ASSERT(n->graph_ == this && !n->inGraphList());
    n->insertBefore(output_);
    return n;
  }

  Node* prependNode(Node* n) {
    ONNX_ASSERT(n->graph_ == this && !n->inGraphList());
    n->insertAfter(output_);
    return n;
  }

  // Adds to graph initializer list, initializer names list, and as a graph input
  // Also syncs the initializer name, tensor name, and value name
  // Create an initializer whose value is stored in input
  Value* addInitializerAndInput(const Tensor& initializer, const std::string& name) {
    Tensor initializerCopy = initializer;
    std::vector<Dimension> dim_sizes{initializerCopy.sizes().cbegin(), initializerCopy.sizes().cend()};
    Value* new_init = addInput();
    initializerCopy.setName(name);
    new_init->setUniqueName(name);
    new_init->setSizes(dim_sizes);
    new_init->setElemType(initializerCopy.elem_type());
    addInitializer(initializerCopy);
    return new_init;
  }

  Value* addInitializerAndInput(const Tensor& initializer) {
    return addInitializerAndInput(initializer, ONNX_NAMESPACE::to_string(getNextUnique()));
  }

  // Erases from graph initializer list, initializer names list, and as a graph input
  // Must have no uses
  void eraseInitializerAndInput(Value* v) {
    eraseInitializer(v->uniqueName());
    if (v->node() == input_) {
      eraseInput(v->offset());
    }
  }

  ~Graph() {
    for (const Node* n : all_nodes)
      delete n;
    for (const Value* v : all_values)
      delete v;
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
  }

  bool has_name() const {
    return has_name_;
  }

  const std::string& name() const {
    return name_;
  }

  void setName(std::string name) {
    has_name_ = true;
    name_ = std::move(name);
  }

  friend std::ostream& operator<<(std::ostream& out, const Graph& g);

  void forSelfAndEachSubGraph(const std::function<void(Graph*)>& fn) {
    fn(this);

    for (const Node* node : all_nodes) {
      for (const auto& attr : node->attributeNames()) {
        if (node->kindOf(attr) == AttributeKind::g) {
          std::shared_ptr<Graph> subgraph = node->g(attr);
          subgraph->forSelfAndEachSubGraph(fn);
        } else if (node->kindOf(attr) == AttributeKind::gs) {
          for (const auto& subgraph : node->gs(attr)) {
            subgraph->forSelfAndEachSubGraph(fn);
          }
        }
      }
    }
  }

  void forSelfAndEachSubGraph(const std::function<void(const Graph*)>& fn) const {
    std::function<void(Graph*)> tmp_fn = [fn](Graph* graph) { fn(graph); };
    const_cast<Graph*>(this)->forSelfAndEachSubGraph(tmp_fn);
  }

  void forEachNode(const std::function<void(Node*)>& fn) {
    forSelfAndEachSubGraph([fn](Graph* graph) {
      for (Node* node : graph->nodes()) {
        fn(node);
      }
    });
  }

  void forEachNode(const std::function<void(const Node*)>& fn) const {
    std::function<void(Node*)> tmp_fn = [fn](Node* node) { fn(node); };
    const_cast<Graph*>(this)->forEachNode(tmp_fn);
  }

 private:
  // should only be called in the constructor
  Node* initOutput(Node* p) {
    p->next() = p;
    p->prev() = p;
    p->setStage(std::numeric_limits<size_t>::max());
    return p;
  }

  void freeNode(Node* n) {
    auto it = all_nodes.find(n);
    ONNX_ASSERT(it != all_nodes.end());
    delete *it;
    all_nodes.erase(it);
  }
  void freeValue(Value* v) {
    auto it = all_values.find(v);
    ONNX_ASSERT(it != all_values.end());
    delete *it;
    all_values.erase(it);
  }
};

inline Value::Value(Node* node_, size_t offset_)
    : node_(node_),
      offset_(offset_),
      unique_(node_->graph_->getNextUnique()),
      stage_(node_->graph_->new_node_stage_),
      has_unique_name_(false),
      elem_type_(ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED),
      has_sizes_(false) {
  node_->graph_->all_values.emplace(this);
}

inline Graph* Value::owningGraph() {
  return node()->owningGraph();
}

inline const Graph* Value::owningGraph() const {
  return node()->owningGraph();
}

// `captured` nodes in subgraph determines which value it captures
// by storing the value's unique name, so old unique names in `captured` nodes
// should also be updated.
// Initializer names are also storaged in graph.initializer_names_, it should be
// updated too.
inline Value* Value::setUniqueName(const std::string& name, bool update_related_names) {
  if (has_unique_name() && update_related_names) {
    auto* graph = owningGraph();
    auto old_name = unique_name_;
    for (size_t i = 0; i < owningGraph()->initializer_names_.size(); i++) {
      auto& initializer_name = owningGraph()->initializer_names_[i];
      if (initializer_name == old_name) {
        initializer_name = name;
        owningGraph()->initializers_[i].setName(name);
      }
    }
    graph->forEachNode([this, &name, &old_name](Node* node) {
      if (node->owningGraph() == this->owningGraph()) {
        // skip non-subgraph
        return;
      }
      if (node->kind() == kCaptured) {
        Value* output = node->output();
        if (output->uniqueName() == old_name) {
          output->setUniqueName(name, false);
        }
      }
    });
  }
  unique_name_ = name;
  has_unique_name_ = true;
  return this;
}

inline void Value::replaceAllUsesWith(Value* newValue) {
  auto* graph = owningGraph();
  ONNX_ASSERT(graph == newValue->owningGraph());
  // propagate sizes and elem type
  if (this->has_sizes()) {
    newValue->setSizes(this->sizes());
  }
  if (this->elemType() != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
    newValue->setElemType(this->elemType());
  }
  const auto unique_name = this->uniqueName();
  // We do not want the optimization to change the graph output name
  if (std::find(graph->outputs().rbegin(), graph->outputs().rend(), this) != graph->outputs().rend()) {
    newValue->setUniqueName(unique_name);
    // The "unique" semantic of unique_name should be kept or uses()
    // will return an incorrect result when the value is used in subgraph
    this->setUniqueName(ONNX_NAMESPACE::to_string(graph->getNextUnique()), false);
  }
  newValue->uses_in_current_graph_.reserve(this->uses_in_current_graph_.size());
  for (auto u : uses_in_current_graph_) {
    u.user->inputs_[u.offset] = newValue;
    newValue->uses_in_current_graph_.push_back(u);
  }
  graph->forEachNode([this, &newValue, &unique_name](Node* node) {
    if (node->owningGraph() == this->owningGraph()) {
      // skip non-subgraph
      return;
    }
    if (node->kind() == kCaptured) {
      Value* output = node->output();
      if (output->uniqueName() == unique_name) {
        output->setUniqueName(newValue->uniqueName());
      }
    }
  });
  uses_in_current_graph_.clear();
  assert(this->uses().empty());
}

inline Node::Node(Graph* graph_, NodeKind kind_)
    : kind_(kind_),
      graph_(graph_),
      stage_(graph_->new_node_stage_),
      has_name_(false),
      has_domain_(false),
      has_doc_string_(false) {
  graph_->all_nodes.emplace(this);
}

inline void Node::eraseOutput(size_t i) {
  ONNX_ASSERT(i < outputs_.size());
  ONNX_ASSERT(outputs_[i]->uses().empty());
  Value* n = outputs_[i];
  outputs_.erase(outputs_.begin() + i);
  owningGraph()->freeValue(n);
  for (size_t j = i; j < outputs_.size(); j++) {
    outputs_[j]->offset_--;
  }
}

inline bool Node::isBefore(Node* n) {
  if (n == nullptr || this == n) {
    // Bail out early.
    return false;
  }
  // return true if node is Param (in initializers)
  if (kind_ == kParam) {
    return true;
  }
  // return false if target node is Param (in initializers)
  if (n->kind() == kParam) {
    return false;
  }
  ONNX_ASSERT(n->inGraphList());
  for (Node* p = next(); p != *graph_->end(); p = p->next()) {
    if (p == n) {
      return true;
    }
  }
  return false;
}

inline void Node::destroy() {
  ONNX_ASSERT(inGraphList());
  while (!outputs().empty())
    eraseOutput(outputs().size() - 1);
  removeAllInputs();
  removeFromList();
  graph_->freeNode(this);
}

/************* All nodes not required to be defined before Graph **************/

inline graph_node_list_iterator Node::iterator() {
  return graph_node_list_iterator(this, 0);
}
inline graph_node_list_iterator Node::reverseIterator() {
  return iterator().reverse();
}
inline const_graph_node_list_iterator Node::iterator() const {
  return const_graph_node_list_iterator(this, 0);
}
inline const_graph_node_list_iterator Node::reverseIterator() const {
  return iterator().reverse();
}

// Returns a list about which nodes are using this value,
// nodes in subgraph are also included.
// This method is usually used to check whether it is
// safe to delete a Value.
inline const use_list Value::uses() const {
  use_list all_uses = uses_in_current_graph_;
  owningGraph()->forEachNode([this, &all_uses](const Node* node) {
    if (node->owningGraph() == this->owningGraph()) {
      // skip non-subgraph
      return;
    }
    if (node->kind() == kCaptured) {
      const Value* output = node->outputs()[0];
      if (output->uniqueName() == this->uniqueName()) {
        const auto output_uses = output->uses();
        all_uses.insert(all_uses.end(), output_uses.begin(), output_uses.end());
      }
    }
  });
  return all_uses;
}

} // namespace ONNX_NAMESPACE
