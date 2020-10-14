#include <string>
#include <vector>
#include <mutex>
#include <storage_credential.h>
#include <curl/curl.h>
#include <rapidjson/document.h>

namespace azure {
    namespace storage_lite {
        const std::string DEFAULT_MSI_URL = "http://169.254.169.254/metadata/identity/oauth2/token";

        std::size_t write_callback(char *in, size_t size, size_t nmemb, std::string *out) {
            std::size_t total_size = size * nmemb;
            if (total_size) {
                out->append(in, total_size);
                return total_size;
            }
            return 0;
        }

        class msi_credential : public storage_credential {
        public:
            AZURE_STORAGE_API msi_credential();

            void sign_request(const storage_request_base &, http_base &, const storage_url &,
                              const storage_headers &) const override;

            void set_msi_url(const std::string &msi_url);

            bool refresh_token(std::string &error) const;

        private:
            std::string m_msi_url;
            mutable std::mutex m_token_mutex;
            mutable time_t m_token_expires;
            mutable std::string m_token;
        };

        AZURE_STORAGE_API msi_credential::msi_credential()
                :m_msi_url(DEFAULT_MSI_URL), m_token_expires(0), m_token("") {
        }

        void msi_credential::sign_request(const storage_request_base &, http_base &h, const storage_url &,
                                          const storage_headers &) const {
            std::string error;
            if (refresh_token(error)) {
                std::string authorization("Bearer ");
                authorization.append(m_token);
                h.add_header(constants::header_authorization, authorization);
            } else {
                std::cerr << "error when refreshing token:" << error << std::endl;
            }
        }

        bool msi_credential::refresh_token(std::string &error) const {
            std::lock_guard <std::mutex> lg(m_token_mutex);
            time_t now = time(0);
            if (m_token_expires != 0 && now < m_token_expires) {
                return true;
            }
            std::string url(m_msi_url);
            url.append("?api-version=2018-02-01&resource=https%3A%2F%2Fstorage.azure.com%2F");
            CURLcode code;
            struct curl_slist *headers = NULL; /* http headers to send with request */
            CURL *ch;
            if ((ch = curl_easy_init()) == NULL) {
                error.assign("curl initialize failed");
                return false;
            }
            headers = curl_slist_append(headers, "metadata: true");
            curl_easy_setopt(ch, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(ch, CURLOPT_URL, url.c_str());
            curl_easy_setopt(ch, CURLOPT_WRITEFUNCTION, write_callback);
            std::string str_buffer;
            curl_easy_setopt(ch, CURLOPT_WRITEDATA, &str_buffer);
            code = curl_easy_perform(ch);

            if (code != CURLE_OK) {
                error.assign("connect to " + url + "failed" );
                curl_easy_cleanup(ch);
                return false;
            }
            long http_code = 0;
            curl_easy_getinfo(ch, CURLINFO_RESPONSE_CODE, &http_code);
            if (http_code >= 200 && http_code < 300) {
                rapidjson::Document d;
                if (d.Parse(str_buffer.c_str()).HasParseError()) {
                    return false;
                }
                if (d.HasMember("access_token")) {
                    m_token.assign(d["access_token"].GetString());
                }
                if (d.HasMember("expires_on")) {
                    auto expires = d["expires_on"].GetString();
                    m_token_expires = std::stol(expires);
                }
                curl_easy_cleanup(ch);
                return true;
            } else {
                error.assign("response code:" + http_code);
                curl_easy_cleanup(ch);
                return false;
            }
        }

        void msi_credential::set_msi_url(const std::string &msi_url) {
            std::lock_guard <std::mutex> lg(m_token_mutex);
            m_msi_url.assign(msi_url);
        }

    } // namespace storage_lite
} // namespace azure
