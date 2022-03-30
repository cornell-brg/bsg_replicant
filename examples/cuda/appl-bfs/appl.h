#ifndef APPL_H
#define APPL_H

template <typename IndexT, typename BodyT>
void parallel_for( IndexT first, IndexT last, IndexT step,
                   const BodyT& body );

template <typename IndexT, typename BodyT>
void parallel_for_1( IndexT first, IndexT last, IndexT step,
                     const BodyT& body );

template <typename IndexT, typename BodyT>
void parallel_for( IndexT first, IndexT last, const BodyT& body );

template <typename IndexT, typename BodyT>
void parallel_for_1( IndexT first, IndexT last, const BodyT& body );

template <typename Func0, typename Func1>
void parallel_invoke( const Func0& func0, const Func1& func1 );

#include "appl.inl"

#endif
