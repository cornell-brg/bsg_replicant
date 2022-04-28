namespace appl {

template <typename IndexT, typename BodyT>
void parallel_for( IndexT first, IndexT last, IndexT step,
                   const BodyT& body )
{
  for (IndexT i = first; i < last; i += step) {
    body(i);
  }
}

template <typename IndexT, typename BodyT>
void parallel_for_1( IndexT first, IndexT last, IndexT step,
                     const BodyT& body ) {
  parallel_for(first, last, step, body);
}

template <typename IndexT, typename BodyT>
void parallel_for( IndexT first, IndexT last, const BodyT& body )
{
  for (IndexT i = first; i < last; i++) {
    body(i);
  }
}

template <typename IndexT, typename BodyT>
void parallel_for_1( IndexT first, IndexT last, const BodyT& body ) {
  parallel_for(first, last, body);
}

template <typename Func0, typename Func1>
void parallel_invoke( const Func0& func0, const Func1& func1 ) {
  func0();
  func1();
}

} // namespace appl
