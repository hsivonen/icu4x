// This file is part of ICU4X. For terms of use, please see the file
// called LICENSE at the top level of the ICU4X source tree
// (online at: https://github.com/unicode-org/icu4x/blob/main/LICENSE ).

use core::iter::FusedIterator;
use core::marker::PhantomData;
use utf8_iter::helpers::bits_to_char;
use utf8_iter::helpers::four_bytes_to_char;
use utf8_iter::helpers::high_ten;
use utf8_iter::helpers::low_five;
use utf8_iter::helpers::low_six;
use utf8_iter::CharIndicesWithHandler;
use utf8_iter::CharsWithHandler;
use utf8_iter::Utf8CharIndicesWithHandler;
use utf8_iter::Utf8CharsEx;
use utf8_iter::Utf8CharsWithHandler;
use utf8_iter::Utf8Handler;

use crate::codepointtrie::AbstractCodePointTrie;
use crate::codepointtrie::TrieValue;

#[derive(Debug)]
pub(crate) struct TrieUtf8Handler<'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    trie: &'trie T,
    phantom: PhantomData<V>,
}

impl<'trie, T, V> TrieUtf8Handler<'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    pub(crate) fn new(trie: &'trie T) -> Self {
        Self {
            trie,
            phantom: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn trie(&self) -> &'trie T {
        self.trie
    }
}

impl<'trie, T, V> Clone for TrieUtf8Handler<'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    fn clone(&self) -> Self {
        Self {
            trie: self.trie,
            phantom: PhantomData,
        }
    }
}

impl<'trie, T, V> Utf8Handler for TrieUtf8Handler<'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    type Output = (char, V);

    /// # Safety-usable invariant
    ///
    /// The trait contract requires the caller to guarantee that `ascii` is ASCII.
    unsafe fn single_byte(&self, ascii: u8) -> Self::Output {
        // SAFETY: The safety-usable invariant from the trait contract is
        // the invariant of `self.trie.ascii`.
        return (char::from(ascii), unsafe { self.trie.ascii(ascii) });
    }

    /// # Safety-usable invariant
    ///
    /// The trait contract requires the caller to guarantee that `first` and `second`
    /// form a two-byte UTF-8 sequence.
    unsafe fn two_byte(&self, first: u8, second: u8) -> Self::Output {
        let high_five = low_five(first);
        let low_six = low_six(second);
        // SAFETY: We've satified the invariants of both `bits_to_char`
        // and `self.trie.utf8_two_byte` by ensuring that the first
        // argument does not have bits other than the low five set
        // and the second argument does not have bits other than
        // the low six set.
        unsafe {
            (
                bits_to_char(high_five, low_six),
                self.trie.utf8_two_byte(high_five, low_six),
            )
        }
    }

    /// # Safety-usable invariant
    ///
    /// The trait contract requires the caller to guarantee that `first`, `second`,
    /// and `third` form a three-byte UTF-8 sequence.
    unsafe fn three_byte(&self, first: u8, second: u8, third: u8) -> Self::Output {
        let high_ten = high_ten(first, second);
        let low_six = low_six(third);
        // SAFETY: We've satified the invariants of both `bits_to_char`
        // and `self.trie.utf8_three_byte` by ensuring that the first
        // argument does not have bits other than the low ten set
        // and the second argument does not have bits other than
        // the low six set and from the safety-usable invariant
        // we know that these bits do not represent a surrogate
        // (for the invariant of `bits_to_char`).
        unsafe {
            (
                bits_to_char(high_ten, low_six),
                self.trie.utf8_three_byte(high_ten, low_six),
            )
        }
    }

    /// # Safety-usable invariant
    ///
    /// The trait contract requires the caller to guarantee that `first`, `second`,
    /// `third`, and `fourth` form a four-byte UTF-8 sequence.
    unsafe fn four_byte(&self, first: u8, second: u8, third: u8, fourth: u8) -> Self::Output {
        // SAFETY: The safety-usable invariant of this method is the invariant of
        // `four_bytes_to_char`.
        let c = unsafe { four_bytes_to_char(first, second, third, fourth) };
        (c, self.trie.supplementary(u32::from(c)))
    }

    fn error(&self) -> Self::Output {
        (
            char::REPLACEMENT_CHARACTER,
            self.trie.bmp(char::REPLACEMENT_CHARACTER as u16),
        )
    }
}

#[derive(Debug)]
pub(crate) struct TrieUtf8HandlerDefaultForAscii<'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    trie: &'trie T,
    phantom: PhantomData<V>,
}

impl<'trie, T, V> TrieUtf8HandlerDefaultForAscii<'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    pub(crate) fn new(trie: &'trie T) -> Self {
        Self {
            trie,
            phantom: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn trie(&self) -> &'trie T {
        self.trie
    }
}

impl<'trie, T, V> Clone for TrieUtf8HandlerDefaultForAscii<'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    fn clone(&self) -> Self {
        Self {
            trie: self.trie,
            phantom: PhantomData,
        }
    }
}

impl<'trie, T, V> Utf8Handler for TrieUtf8HandlerDefaultForAscii<'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    type Output = (char, V);

    /// # Safety-usable invariant
    ///
    /// The trait contract requires the caller to guarantee that `ascii` is ASCII.
    unsafe fn single_byte(&self, ascii: u8) -> Self::Output {
        // SAFETY: The safety-usable invariant from the trait contract is
        // the invariant of `self.trie.ascii`.
        return (char::from(ascii), V::default());
    }

    /// # Safety-usable invariant
    ///
    /// The trait contract requires the caller to guarantee that `first` and `second`
    /// form a two-byte UTF-8 sequence.
    unsafe fn two_byte(&self, first: u8, second: u8) -> Self::Output {
        let high_five = low_five(first);
        let low_six = low_six(second);
        // SAFETY: We've satified the invariants of both `bits_to_char`
        // and `self.trie.utf8_two_byte` by ensuring that the first
        // argument does not have bits other than the low five set
        // and the second argument does not have bits other than
        // the low six set.
        unsafe {
            (
                bits_to_char(high_five, low_six),
                self.trie.utf8_two_byte(high_five, low_six),
            )
        }
    }

    /// # Safety-usable invariant
    ///
    /// The trait contract requires the caller to guarantee that `first`, `second`,
    /// and `third` form a three-byte UTF-8 sequence.
    unsafe fn three_byte(&self, first: u8, second: u8, third: u8) -> Self::Output {
        let high_ten = high_ten(first, second);
        let low_six = low_six(third);
        // SAFETY: We've satified the invariants of both `bits_to_char`
        // and `self.trie.utf8_three_byte` by ensuring that the first
        // argument does not have bits other than the low ten set
        // and the second argument does not have bits other than
        // the low six set and from the safety-usable invariant
        // we know that these bits do not represent a surrogate
        // (for the invariant of `bits_to_char`).
        unsafe {
            (
                bits_to_char(high_ten, low_six),
                self.trie.utf8_three_byte(high_ten, low_six),
            )
        }
    }

    /// # Safety-usable invariant
    ///
    /// The trait contract requires the caller to guarantee that `first`, `second`,
    /// `third`, and `fourth` form a four-byte UTF-8 sequence.
    unsafe fn four_byte(&self, first: u8, second: u8, third: u8, fourth: u8) -> Self::Output {
        // SAFETY: The safety-usable invariant of this method is the invariant of
        // `four_bytes_to_char`.
        let c = unsafe { four_bytes_to_char(first, second, third, fourth) };
        (c, self.trie.supplementary(u32::from(c)))
    }

    fn error(&self) -> Self::Output {
        (
            char::REPLACEMENT_CHARACTER,
            self.trie.bmp(char::REPLACEMENT_CHARACTER as u16),
        )
    }
}

/// Provides a trie accessor for types (likely iterators)
/// that are holding a reference to a type that implements
/// `AbstractCodePointTrie`.
pub trait WithTrie<'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Get a reference to the trie.
    fn trie(&self) -> &'trie T;
}

// ---

/// Iterator over `str` by `char` and `TrieValue`.
#[derive(Debug)]
pub struct CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    delegate: CharsWithHandler<'slice, TrieUtf8Handler<'trie, T, V>>,
}

impl<'slice, 'trie, T, V> CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Construct a new `CharsWithTrie`.
    #[inline]
    pub fn new(s: &'slice str, trie: &'trie T) -> Self {
        Self {
            delegate: CharsWithHandler::new(s, TrieUtf8Handler::new(trie)),
        }
    }

    /// Obtains the remainder of the iterator as a string slice.
    #[inline]
    pub fn as_str(&self) -> &'slice str {
        self.delegate.as_str()
    }
}

impl<'slice, 'trie, T, V> Clone for CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    fn clone(&self) -> Self {
        Self {
            delegate: self.delegate.clone(),
        }
    }
}

impl<'slice, 'trie, T, V> WithTrie<'trie, T, V> for CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn trie(&self) -> &'trie T {
        self.delegate.handler().trie()
    }
}

impl<'slice, 'trie, T, V> Iterator for CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    type Item = (char, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.delegate.next()
    }

    #[inline]
    fn count(self) -> usize {
        self.as_str().chars().count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.as_str().chars().size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    // TODO: Delegate advance_by to `Chars` once stabilized.
}

impl<'slice, 'trie, T, V> DoubleEndedIterator for CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.delegate.next_back()
    }
}

impl<'slice, 'trie, T, V> FusedIterator for CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
}
// --

/// Iterator over `str` by `char` and `TrieValue`.
#[derive(Debug)]
pub struct CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    delegate: CharIndicesWithHandler<'slice, TrieUtf8Handler<'trie, T, V>>,
}

impl<'slice, 'trie, T, V> CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Construct a new `CharIndicesWithTrie`.
    #[inline]
    pub fn new(s: &'slice str, trie: &'trie T) -> Self {
        Self {
            delegate: CharIndicesWithHandler::new(s, TrieUtf8Handler::new(trie)),
        }
    }

    /// Obtains the remainder of the iterator as a string slice.
    #[inline]
    pub fn as_str(&self) -> &'slice str {
        self.delegate.as_str()
    }
}

impl<'slice, 'trie, T, V> Clone for CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            delegate: self.delegate.clone(),
        }
    }
}

impl<'slice, 'trie, T, V> WithTrie<'trie, T, V> for CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn trie(&self) -> &'trie T {
        self.delegate.handler().trie()
    }
}

impl<'slice, 'trie, T, V> Iterator for CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    type Item = (usize, char, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (i, (c, v)) = self.delegate.next()?;
        Some((i, c, v))
    }

    #[inline]
    fn count(self) -> usize {
        self.as_str().chars().count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.as_str().chars().size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        // XXX: Is this correct when it doesn't change the internal state as consumed?
        self.next_back()
    }

    // TODO: Delegate advance_by to `Chars` once stabilized.
}

impl<'slice, 'trie, T, V> DoubleEndedIterator for CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let (i, (c, v)) = self.delegate.next_back()?;
        Some((i, c, v))
    }
}

impl<'slice, 'trie, T, V> FusedIterator for CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
}

// --

/// Adds convenience methods to `str`.
pub trait CharsWithTrieEx<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Method for easily creating `CharsWithTrie` on `str` analogously to `chars()`.
    fn chars_with_trie(&'slice self, trie: &'trie T) -> CharsWithTrie<'slice, 'trie, T, V>;

    /// Method for easily creating `CharIndicesWithTrie` on `str` analogously to `char_indices()`.
    fn char_indices_with_trie(
        &'slice self,
        trie: &'trie T,
    ) -> CharIndicesWithTrie<'slice, 'trie, T, V>;
}

impl<'slice, 'trie, T, V> CharsWithTrieEx<'slice, 'trie, T, V> for str
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Method for easily creating `CharsWithTrie` on `str` analogously to `chars()`.
    #[inline]
    fn chars_with_trie(&'slice self, trie: &'trie T) -> CharsWithTrie<'slice, 'trie, T, V> {
        CharsWithTrie::new(self, trie)
    }

    /// Method for easily creating `CharIndicesWithTrie` on `str` analogously to `char_indices()`.
    #[inline]
    fn char_indices_with_trie(
        &'slice self,
        trie: &'trie T,
    ) -> CharIndicesWithTrie<'slice, 'trie, T, V> {
        CharIndicesWithTrie::new(self, trie)
    }
}

// --

/// Iterator over `str` by `char` and `TrieValue` but
/// the trie value for ASCII is `V::default()` instead of
/// reading from the trie. (`V::default()` can be optimized
/// on at compile time while reading the trie's default value
/// is a run-time operation.)
#[derive(Debug)]
pub struct CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    delegate: CharsWithHandler<'slice, TrieUtf8HandlerDefaultForAscii<'trie, T, V>>,
}

impl<'slice, 'trie, T, V> CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Construct a new `CharsWithTrieDefaultForAscii`.
    #[inline]
    pub fn new(s: &'slice str, trie: &'trie T) -> Self {
        Self {
            delegate: CharsWithHandler::new(s, TrieUtf8HandlerDefaultForAscii::new(trie)),
        }
    }

    /// Obtains the remainder of the iterator as a string slice.
    #[inline]
    pub fn as_str(&self) -> &'slice str {
        self.delegate.as_str()
    }
}

impl<'slice, 'trie, T, V> Clone for CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    fn clone(&self) -> Self {
        Self {
            delegate: self.delegate.clone(),
        }
    }
}

impl<'slice, 'trie, T, V> WithTrie<'trie, T, V>
    for CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn trie(&self) -> &'trie T {
        self.delegate.handler().trie()
    }
}

impl<'slice, 'trie, T, V> Iterator for CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    type Item = (char, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.delegate.next()
    }

    #[inline]
    fn count(self) -> usize {
        self.as_str().chars().count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.as_str().chars().size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    // TODO: Delegate advance_by to `Chars` once stabilized.
}

impl<'slice, 'trie, T, V> DoubleEndedIterator for CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.delegate.next_back()
    }
}

impl<'slice, 'trie, T, V> FusedIterator for CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
}
// --

/// Iterator over `str` by `char` and `TrieValue`.
#[derive(Debug)]
pub struct CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    delegate: CharIndicesWithHandler<'slice, TrieUtf8HandlerDefaultForAscii<'trie, T, V>>,
}

impl<'slice, 'trie, T, V> CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Construct a new `CharIndicesWithTrieDefaultForAscii`.
    #[inline]
    pub fn new(s: &'slice str, trie: &'trie T) -> Self {
        Self {
            delegate: CharIndicesWithHandler::new(s, TrieUtf8HandlerDefaultForAscii::new(trie)),
        }
    }

    /// Obtains the remainder of the iterator as a string slice.
    #[inline]
    pub fn as_str(&self) -> &'slice str {
        self.delegate.as_str()
    }
}

impl<'slice, 'trie, T, V> Clone for CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            delegate: self.delegate.clone(),
        }
    }
}

impl<'slice, 'trie, T, V> WithTrie<'trie, T, V>
    for CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn trie(&self) -> &'trie T {
        self.delegate.handler().trie()
    }
}

impl<'slice, 'trie, T, V> Iterator for CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    type Item = (usize, char, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (i, (c, v)) = self.delegate.next()?;
        Some((i, c, v))
    }

    #[inline]
    fn count(self) -> usize {
        self.as_str().chars().count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.as_str().chars().size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        // XXX: Is this correct when it doesn't change the internal state as consumed?
        self.next_back()
    }

    // TODO: Delegate advance_by to `Chars` once stabilized.
}

impl<'slice, 'trie, T, V> DoubleEndedIterator
    for CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let (i, (c, v)) = self.delegate.next_back()?;
        Some((i, c, v))
    }
}

impl<'slice, 'trie, T, V> FusedIterator for CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
}

// --

/// Adds convenience methods to `str`.
pub trait CharsWithTrieDefaultForAsciiEx<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Method for easily creating `CharsWithTrie` on `str` analogously to `chars()`.
    fn chars_with_trie_default_for_ascii(
        &'slice self,
        trie: &'trie T,
    ) -> CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>;

    /// Method for easily creating `CharIndicesWithTrie` on `str` analogously to `char_indices()`.
    fn char_indices_with_trie_default_for_ascii(
        &'slice self,
        trie: &'trie T,
    ) -> CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>;
}

impl<'slice, 'trie, T, V> CharsWithTrieDefaultForAsciiEx<'slice, 'trie, T, V> for str
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Method for easily creating `CharsWithTrie` on `str` analogously to `chars()`.
    #[inline]
    fn chars_with_trie_default_for_ascii(
        &'slice self,
        trie: &'trie T,
    ) -> CharsWithTrieDefaultForAscii<'slice, 'trie, T, V> {
        CharsWithTrieDefaultForAscii::new(self, trie)
    }

    /// Method for easily creating `CharIndicesWithTrie` on `str` analogously to `char_indices()`.
    #[inline]
    fn char_indices_with_trie_default_for_ascii(
        &'slice self,
        trie: &'trie T,
    ) -> CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V> {
        CharIndicesWithTrieDefaultForAscii::new(self, trie)
    }
}

// --

/// Iterator over `str` by `char` and `TrieValue`.
#[derive(Debug)]
pub struct Utf8CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    delegate: Utf8CharsWithHandler<'slice, TrieUtf8Handler<'trie, T, V>>,
}

impl<'slice, 'trie, T, V> Utf8CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Construct a new `Utf8CharsWithTrie`.
    #[inline]
    pub fn new(bytes: &'slice [u8], trie: &'trie T) -> Self {
        Self {
            delegate: Utf8CharsWithHandler::new(bytes, TrieUtf8Handler::new(trie)),
        }
    }

    /// Obtains the remainder of the iterator as a string slice.
    #[inline]
    pub fn as_slice(&self) -> &'slice [u8] {
        self.delegate.as_slice()
    }
}

impl<'slice, 'trie, T, V> Clone for Utf8CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    fn clone(&self) -> Self {
        Self {
            delegate: self.delegate.clone(),
        }
    }
}

impl<'slice, 'trie, T, V> WithTrie<'trie, T, V> for Utf8CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn trie(&self) -> &'trie T {
        self.delegate.handler().trie()
    }
}

impl<'slice, 'trie, T, V> Iterator for Utf8CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    type Item = (char, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.delegate.next()
    }

    #[inline]
    fn count(self) -> usize {
        self.as_slice().chars().count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.as_slice().chars().size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    // TODO: Delegate advance_by to `Chars` once stabilized.
}

impl<'slice, 'trie, T, V> DoubleEndedIterator for Utf8CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.delegate.next_back()
    }
}

impl<'slice, 'trie, T, V> FusedIterator for Utf8CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
}
// --

/// Iterator over `str` by `char` and `TrieValue`.
#[derive(Debug)]
pub struct Utf8CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    delegate: Utf8CharIndicesWithHandler<'slice, TrieUtf8Handler<'trie, T, V>>,
}

impl<'slice, 'trie, T, V> Utf8CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Construct a new `Utf8CharIndicesWithTrie`.
    #[inline]
    pub fn new(bytes: &'slice [u8], trie: &'trie T) -> Self {
        Self {
            delegate: Utf8CharIndicesWithHandler::new(bytes, TrieUtf8Handler::new(trie)),
        }
    }

    /// Obtains the remainder of the iterator as a string slice.
    #[inline]
    pub fn as_slice(&self) -> &'slice [u8] {
        self.delegate.as_slice()
    }
}

impl<'slice, 'trie, T, V> Clone for Utf8CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            delegate: self.delegate.clone(),
        }
    }
}

impl<'slice, 'trie, T, V> WithTrie<'trie, T, V> for Utf8CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn trie(&self) -> &'trie T {
        self.delegate.handler().trie()
    }
}

impl<'slice, 'trie, T, V> Iterator for Utf8CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    type Item = (usize, char, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (i, (c, v)) = self.delegate.next()?;
        Some((i, c, v))
    }

    #[inline]
    fn count(self) -> usize {
        self.as_slice().chars().count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.as_slice().chars().size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        // XXX: Is this correct when it doesn't change the internal state as consumed?
        self.next_back()
    }

    // TODO: Delegate advance_by to `Chars` once stabilized.
}

impl<'slice, 'trie, T, V> DoubleEndedIterator for Utf8CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let (i, (c, v)) = self.delegate.next_back()?;
        Some((i, c, v))
    }
}

impl<'slice, 'trie, T, V> FusedIterator for Utf8CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
}

// --

/// Adds convenience methods to `&[u8]`.
pub trait Utf8CharsWithTrieEx<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Method for easily creating `Utf8CharsWithTrie` on `str` analogously to `chars()`.
    fn chars_with_trie(&'slice self, trie: &'trie T) -> Utf8CharsWithTrie<'slice, 'trie, T, V>;

    /// Method for easily creating `Utf8CharIndicesWithTrie` on `str` analogously to `char_indices()`.
    fn char_indices_with_trie(
        &'slice self,
        trie: &'trie T,
    ) -> Utf8CharIndicesWithTrie<'slice, 'trie, T, V>;
}

impl<'slice, 'trie, T, V> Utf8CharsWithTrieEx<'slice, 'trie, T, V> for [u8]
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Method for easily creating `Utf8CharsWithTrie` on `str` analogously to `chars()`.
    #[inline]
    fn chars_with_trie(&'slice self, trie: &'trie T) -> Utf8CharsWithTrie<'slice, 'trie, T, V> {
        Utf8CharsWithTrie::new(self, trie)
    }

    /// Method for easily creating `Utf8CharIndicesWithTrie` on `str` analogously to `char_indices()`.
    #[inline]
    fn char_indices_with_trie(
        &'slice self,
        trie: &'trie T,
    ) -> Utf8CharIndicesWithTrie<'slice, 'trie, T, V> {
        Utf8CharIndicesWithTrie::new(self, trie)
    }
}

// --

/// Iterator over `str` by `char` and `TrieValue` but
/// the trie value for ASCII is `V::default()` instead of
/// reading from the trie. (`V::default()` can be optimized
/// on at compile time while reading the trie's default value
/// is a run-time operation.)
#[derive(Debug)]
pub struct Utf8CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    delegate: Utf8CharsWithHandler<'slice, TrieUtf8HandlerDefaultForAscii<'trie, T, V>>,
}

impl<'slice, 'trie, T, V> Utf8CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Construct a new `Utf8CharsWithTrieDefaultForAscii`.
    #[inline]
    pub fn new(bytes: &'slice [u8], trie: &'trie T) -> Self {
        Self {
            delegate: Utf8CharsWithHandler::new(bytes, TrieUtf8HandlerDefaultForAscii::new(trie)),
        }
    }

    /// Obtains the remainder of the iterator as a string slice.
    #[inline]
    pub fn as_slice(&self) -> &'slice [u8] {
        self.delegate.as_slice()
    }
}

impl<'slice, 'trie, T, V> Clone for Utf8CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    fn clone(&self) -> Self {
        Self {
            delegate: self.delegate.clone(),
        }
    }
}

impl<'slice, 'trie, T, V> WithTrie<'trie, T, V>
    for Utf8CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn trie(&self) -> &'trie T {
        self.delegate.handler().trie()
    }
}

impl<'slice, 'trie, T, V> Iterator for Utf8CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    type Item = (char, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.delegate.next()
    }

    #[inline]
    fn count(self) -> usize {
        self.as_slice().chars().count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.as_slice().chars().size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    // TODO: Delegate advance_by to `Chars` once stabilized.
}

impl<'slice, 'trie, T, V> DoubleEndedIterator
    for Utf8CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.delegate.next_back()
    }
}

impl<'slice, 'trie, T, V> FusedIterator for Utf8CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
}
// --

/// Iterator over `str` by `char` and `TrieValue`.
#[derive(Debug)]
pub struct Utf8CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    delegate: Utf8CharIndicesWithHandler<'slice, TrieUtf8HandlerDefaultForAscii<'trie, T, V>>,
}

impl<'slice, 'trie, T, V> Utf8CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Construct a new `Utf8CharIndicesWithTrieDefaultForAscii`.
    #[inline]
    pub fn new(bytes: &'slice [u8], trie: &'trie T) -> Self {
        Self {
            delegate: Utf8CharIndicesWithHandler::new(
                bytes,
                TrieUtf8HandlerDefaultForAscii::new(trie),
            ),
        }
    }

    /// Obtains the remainder of the iterator as a string slice.
    #[inline]
    pub fn as_slice(&self) -> &'slice [u8] {
        self.delegate.as_slice()
    }
}

impl<'slice, 'trie, T, V> Clone for Utf8CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            delegate: self.delegate.clone(),
        }
    }
}

impl<'slice, 'trie, T, V> WithTrie<'trie, T, V>
    for Utf8CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn trie(&self) -> &'trie T {
        self.delegate.handler().trie()
    }
}

impl<'slice, 'trie, T, V> Iterator for Utf8CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    type Item = (usize, char, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (i, (c, v)) = self.delegate.next()?;
        Some((i, c, v))
    }

    #[inline]
    fn count(self) -> usize {
        self.as_slice().chars().count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.as_slice().chars().size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        // XXX: Is this correct when it doesn't change the internal state as consumed?
        self.next_back()
    }

    // TODO: Delegate advance_by to `Chars` once stabilized.
}

impl<'slice, 'trie, T, V> DoubleEndedIterator
    for Utf8CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let (i, (c, v)) = self.delegate.next_back()?;
        Some((i, c, v))
    }
}

impl<'slice, 'trie, T, V> FusedIterator
    for Utf8CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
}

// --

/// Adds convenience methods to `[u8]`.
pub trait Utf8CharsWithTrieDefaultForAsciiEx<'slice, 'trie, T, V>
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Method for easily creating `Utf8CharsWithTrie` on `str` analogously to `chars()`.
    fn chars_with_trie_default_for_ascii(
        &'slice self,
        trie: &'trie T,
    ) -> Utf8CharsWithTrieDefaultForAscii<'slice, 'trie, T, V>;

    /// Method for easily creating `Utf8CharIndicesWithTrie` on `str` analogously to `char_indices()`.
    fn char_indices_with_trie_default_for_ascii(
        &'slice self,
        trie: &'trie T,
    ) -> Utf8CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V>;
}

impl<'slice, 'trie, T, V> Utf8CharsWithTrieDefaultForAsciiEx<'slice, 'trie, T, V> for [u8]
where
    V: TrieValue + Default,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Method for easily creating `Utf8CharsWithTrie` on `str` analogously to `chars()`.
    #[inline]
    fn chars_with_trie_default_for_ascii(
        &'slice self,
        trie: &'trie T,
    ) -> Utf8CharsWithTrieDefaultForAscii<'slice, 'trie, T, V> {
        Utf8CharsWithTrieDefaultForAscii::new(self, trie)
    }

    /// Method for easily creating `Utf8CharIndicesWithTrie` on `str` analogously to `char_indices()`.
    #[inline]
    fn char_indices_with_trie_default_for_ascii(
        &'slice self,
        trie: &'trie T,
    ) -> Utf8CharIndicesWithTrieDefaultForAscii<'slice, 'trie, T, V> {
        Utf8CharIndicesWithTrieDefaultForAscii::new(self, trie)
    }
}

// --

/// Iterator over Latin1 `[u8]` by `char` and `TrieValue`.
#[derive(Debug)]
pub struct Latin1CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    delegate: core::slice::Iter<'slice, u8>,
    trie: &'trie T,
    phantom: PhantomData<V>,
}

impl<'slice, 'trie, T, V> Latin1CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Construct a new `Latin1CharsWithTrie`.
    #[inline]
    pub fn new(s: &'slice [u8], trie: &'trie T) -> Self {
        Self {
            delegate: s.iter(),
            trie,
            phantom: PhantomData,
        }
    }

    /// Obtains the remainder of the iterator as a slice.
    #[inline]
    pub fn as_slice(&self) -> &'slice [u8] {
        self.delegate.as_slice()
    }
}

impl<'slice, 'trie, T, V> Clone for Latin1CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            delegate: self.delegate.clone(),
            trie: self.trie,
            phantom: PhantomData,
        }
    }
}

impl<'slice, 'trie, T, V> WithTrie<'trie, T, V> for Latin1CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn trie(&self) -> &'trie T {
        self.trie
    }
}

impl<'slice, 'trie, T, V> Iterator for Latin1CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    type Item = (char, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let b = *self.delegate.next()?;
        Some((char::from(b), self.trie.latin1(b)))
    }

    #[inline]
    fn count(self) -> usize {
        self.delegate.count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.delegate.size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    // TODO: Delegate advance_by to `delegate` once stabilized.
}

impl<'slice, 'trie, T, V> DoubleEndedIterator for Latin1CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let b = *self.delegate.next_back()?;
        Some((char::from(b), self.trie.latin1(b)))
    }
}

impl<'slice, 'trie, T, V> FusedIterator for Latin1CharsWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
}

// --

/// Iterator over `str` by `char` and `TrieValue`.
#[derive(Debug)]
pub struct Latin1CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    offset: usize,
    delegate: core::slice::Iter<'slice, u8>,
    trie: &'trie T,
    phantom: PhantomData<V>,
}

impl<'slice, 'trie, T, V> Latin1CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Construct a new `Latin1CharIndicesWithTrie`.
    #[inline]
    pub fn new(s: &'slice [u8], trie: &'trie T) -> Self {
        Self {
            offset: 0,
            delegate: s.iter(),
            trie,
            phantom: PhantomData,
        }
    }

    /// Obtains the remainder of the iterator as a slice.
    #[inline]
    pub fn as_slice(&self) -> &'slice [u8] {
        self.delegate.as_slice()
    }
}

impl<'slice, 'trie, T, V> Clone for Latin1CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            offset: self.offset,
            delegate: self.delegate.clone(),
            trie: self.trie,
            phantom: PhantomData,
        }
    }
}

impl<'slice, 'trie, T, V> WithTrie<'trie, T, V> for Latin1CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn trie(&self) -> &'trie T {
        self.trie
    }
}

impl<'slice, 'trie, T, V> Iterator for Latin1CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    type Item = (usize, char, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let b = *self.delegate.next()?;
        let old_offset = self.offset;
        self.offset += 1;
        Some((old_offset, char::from(b), self.trie.latin1(b)))
    }

    #[inline]
    fn count(self) -> usize {
        self.delegate.count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.delegate.size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    // TODO: Delegate advance_by to `delegate` once stabilized.
}

impl<'slice, 'trie, T, V> DoubleEndedIterator for Latin1CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let b = *self.delegate.next_back()?;
        Some((
            self.offset + self.as_slice().len(),
            char::from(b),
            self.trie.latin1(b),
        ))
    }
}

impl<'slice, 'trie, T, V> FusedIterator for Latin1CharIndicesWithTrie<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
}

// --

/// Adds convenience methods to `[u8]`.
pub trait Latin1CharsWithTrieEx<'slice, 'trie, T, V>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Method for easily creating `Latin1CharsWithTrie` on `[u8]` analogously to `chars()` on `str`.
    /// (The name is prefixed with `latin1_` to avoid ambiguity with interpreting [u8] as UTF-8.)
    fn latin1_chars_with_trie(
        &'slice self,
        trie: &'trie T,
    ) -> Latin1CharsWithTrie<'slice, 'trie, T, V>;

    /// Method for easily creating `Latin1CharIndicesWithTrie` on `str` analogously to `char_indices()` on `str`.
    /// (The name is prefixed with `latin1_` to avoid ambiguity with interpreting [u8] as UTF-8.)
    fn latin1_char_indices_with_trie(
        &'slice self,
        trie: &'trie T,
    ) -> Latin1CharIndicesWithTrie<'slice, 'trie, T, V>;
}

impl<'slice, 'trie, T, V> Latin1CharsWithTrieEx<'slice, 'trie, T, V> for [u8]
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
{
    /// Method for easily creating `Latin1CharsWithTrie` on `[u8]` analogously to `chars()` on `str`.
    /// (The name is prefixed with `latin1_` to avoid ambiguity with interpreting [u8] as UTF-8.)
    #[inline]
    fn latin1_chars_with_trie(
        &'slice self,
        trie: &'trie T,
    ) -> Latin1CharsWithTrie<'slice, 'trie, T, V> {
        Latin1CharsWithTrie::new(self, trie)
    }

    /// Method for easily creating `Latin1CharIndicesWithTrie` on `str` analogously to `char_indices()` on `str`.
    /// (The name is prefixed with `latin1_` to avoid ambiguity with interpreting [u8] as UTF-8.)
    #[inline]
    fn latin1_char_indices_with_trie(
        &'slice self,
        trie: &'trie T,
    ) -> Latin1CharIndicesWithTrie<'slice, 'trie, T, V> {
        Latin1CharIndicesWithTrie::new(self, trie)
    }
}

// --

/// Wraps an `Iterator<Item = char>` with a reference to
/// an `AbstractCodePointTrie`.
#[derive(Debug)]
pub struct CharIterWithTrie<'trie, T, V, I>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
    I: Iterator<Item = char>,
{
    delegate: I,
    trie: &'trie T,
    phantom: PhantomData<V>,
}

impl<'trie, T, V, I> CharIterWithTrie<'trie, T, V, I>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
    I: Iterator<Item = char>,
{
    /// Constructs a new `CharIterWithTrie`.
    #[inline]
    pub fn new(iter: I, trie: &'trie T) -> Self {
        Self {
            delegate: iter,
            trie,
            phantom: PhantomData,
        }
    }
}

impl<'trie, T, V, I> WithTrie<'trie, T, V> for CharIterWithTrie<'trie, T, V, I>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
    I: Iterator<Item = char>,
{
    #[inline]
    fn trie(&self) -> &'trie T {
        self.trie
    }
}

impl<'trie, T, V, I> Iterator for CharIterWithTrie<'trie, T, V, I>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
    I: Iterator<Item = char>,
{
    type Item = (char, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let c = self.delegate.next()?;
        Some((c, self.trie.scalar(c)))
    }

    #[inline]
    fn count(self) -> usize {
        self.delegate.count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.delegate.size_hint()
    }

    // Looks like conditionally implementing `last()` is not allowed.

    // TODO: Delegate advance_by to `delegate` once stabilized.
}

impl<'trie, T, V, I> DoubleEndedIterator for CharIterWithTrie<'trie, T, V, I>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
    I: DoubleEndedIterator<Item = char>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let c = self.delegate.next_back()?;
        Some((c, self.trie.scalar(c)))
    }
}

impl<'trie, T, V, I> FusedIterator for CharIterWithTrie<'trie, T, V, I>
where
    V: TrieValue,
    T: AbstractCodePointTrie<'trie, V>,
    I: FusedIterator<Item = char>,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward() {
        let trie = crate::codepointtrie::planes::get_planes_trie();
        let s = "abäαあ🥳𧉧";
        let mut iter = s.chars_with_trie(&trie);
        assert_eq!(iter.next(), Some(('a', 0)));
        assert_eq!(iter.next(), Some(('b', 0)));
        assert_eq!(iter.next(), Some(('ä', 0)));
        assert_eq!(iter.next(), Some(('α', 0)));
        assert_eq!(iter.next(), Some(('あ', 0)));
        assert_eq!(iter.next(), Some(('🥳', 1)));
        assert_eq!(iter.next(), Some(('𧉧', 2)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_backwards() {
        let trie = crate::codepointtrie::planes::get_planes_trie();
        let s = "abäαあ🥳𧉧";
        let mut iter = s.chars_with_trie(&trie);
        assert_eq!(iter.next_back(), Some(('𧉧', 2)));
        assert_eq!(iter.next_back(), Some(('🥳', 1)));
        assert_eq!(iter.next_back(), Some(('あ', 0)));
        assert_eq!(iter.next_back(), Some(('α', 0)));
        assert_eq!(iter.next_back(), Some(('ä', 0)));
        assert_eq!(iter.next_back(), Some(('b', 0)));
        assert_eq!(iter.next_back(), Some(('a', 0)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_indices_forward() {
        let trie = crate::codepointtrie::planes::get_planes_trie();
        let s = "abäαあ🥳𧉧";
        let mut iter = s.char_indices_with_trie(&trie);
        assert_eq!(iter.next(), Some((0, 'a', 0)));
        assert_eq!(iter.next(), Some((1, 'b', 0)));
        assert_eq!(iter.next(), Some((2, 'ä', 0)));
        assert_eq!(iter.next(), Some((4, 'α', 0)));
        assert_eq!(iter.next(), Some((6, 'あ', 0)));
        assert_eq!(iter.next(), Some((9, '🥳', 1)));
        assert_eq!(iter.next(), Some((13, '𧉧', 2)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_indices_backwards() {
        let trie = crate::codepointtrie::planes::get_planes_trie();
        let s = "abäαあ🥳𧉧";
        let mut iter = s.char_indices_with_trie(&trie);
        assert_eq!(iter.next_back(), Some((13, '𧉧', 2)));
        assert_eq!(iter.next_back(), Some((9, '🥳', 1)));
        assert_eq!(iter.next_back(), Some((6, 'あ', 0)));
        assert_eq!(iter.next_back(), Some((4, 'α', 0)));
        assert_eq!(iter.next_back(), Some((2, 'ä', 0)));
        assert_eq!(iter.next_back(), Some((1, 'b', 0)));
        assert_eq!(iter.next_back(), Some((0, 'a', 0)));
        assert_eq!(iter.next(), None);
    }
}
