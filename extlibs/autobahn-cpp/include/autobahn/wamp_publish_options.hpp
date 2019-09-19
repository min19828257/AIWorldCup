///////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) Crossbar.io Technologies GmbH and contributors
//
// Boost Software License - Version 1.0 - August 17th, 2003
//
// Permission is hereby granted, free of charge, to any person or organization
// obtaining a copy of the software and accompanying documentation covered by
// this license (the "Software") to use, reproduce, display, distribute,
// execute, and transmit the Software, and to prepare derivative works of the
// Software, and to permit third-parties to whom the Software is furnished to
// do so, all subject to the following:
//
// The copyright notices in the Software and this entire statement, including
// the above license grant, this restriction and the following disclaimer,
// must be included in all copies of the Software, in whole or in part, and
// all derivative works of the Software, unless such copies or derivative
// works are solely in the form of machine-executable object code generated by
// a source language processor.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
// SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
// FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef AUTOBAHN_WAMP_PUBLISH_OPTIONS_HPP
#define AUTOBAHN_WAMP_PUBLISH_OPTIONS_HPP

#include <chrono>

namespace autobahn {

/*!
 * \ingroup PUB
 * Options for publishing.
 */
class wamp_publish_options
{
public:
    wamp_publish_options();

    wamp_publish_options(wamp_publish_options&& other) = delete;
    wamp_publish_options(const wamp_publish_options& other) = delete;
    wamp_publish_options& operator=(wamp_publish_options&& other) = delete;
    wamp_publish_options& operator=(const wamp_publish_options& other) = delete;


    const bool& exclude_me() const;

    void set_exclude_me(const bool& exclude_me);

private:
    bool m_exclude_me;
};

} // namespace autobahn

#include "wamp_publish_options.ipp"

#endif // AUTOBAHN_WAMP_PUBLISH_OPTIONS_HPP
